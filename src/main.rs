use std::{
    env, str::FromStr, sync::Arc, time::Instant
};

use atomic_float::AtomicF64;
use axum::{extract::Path, routing::get, Router};
use cudarc::{
    driver::{CudaDevice, CudaSlice},
    nccl::{
        Comm, Id,
    },
};
use once_cell::sync::Lazy;
use tokio::{sync::Barrier, task::JoinHandle};
use std::sync::atomic::Ordering::{Acquire, SeqCst};

static COMM_ID: Lazy<Vec<Id>> = Lazy::new(|| {
    (0..CudaDevice::count().unwrap())
        .map(|_| Id::new().unwrap())
        .collect::<Vec<_>>()
});

struct IdWrapper(Id);

impl FromStr for IdWrapper {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let bytes = hex::decode(s)
            .unwrap()
            .iter()
            .map(|&c| c as i8)
            .collect::<Vec<_>>();

        let mut id = [0i8; 128];
        id.copy_from_slice(&bytes);

        Ok(IdWrapper(Id::uninit(id)))
    }
}

impl ToString for IdWrapper {
    fn to_string(&self) -> String {
        hex::encode(
            self.0
                .internal()
                .iter()
                .map(|&c| c as u8)
                .collect::<Vec<_>>(),
        )
    }
}

const DUMMY_DATA_LEN: usize = 35 * (1 << 30);

async fn root(Path(device_id): Path<String>) -> String {
    let device_id: usize = device_id.parse().unwrap();
    IdWrapper(COMM_ID[device_id]).to_string()
}

#[tokio::main]
async fn main() -> eyre::Result<()> {
    let args = env::args().collect::<Vec<_>>();
    let n_devices = CudaDevice::count().unwrap() as usize;
    let party_id: usize = args[1].parse().unwrap();
    let total_throughput = Arc::new(AtomicF64::new(0.0));

    if party_id == 0 {
        tokio::spawn(async move {
            let app = Router::new().route("/:device_id", get(root));

            let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
            axum::serve(listener, app).await.unwrap();
        });
    };

    let mut devices: Vec<Arc<CudaDevice>> = vec![];
    let mut slices: Vec<CudaSlice<u8>> = vec![];
    for i in 0..n_devices {
        let dev = CudaDevice::new(i).unwrap();
        let mut slice: CudaSlice<u8> = dev.alloc_zeros(DUMMY_DATA_LEN).unwrap();
        dev.synchronize();
        slices.push(slice);
        devices.push(dev);
    }
    
    let barrier = Arc::new(Barrier::new(n_devices));
    let mut handles: Vec<JoinHandle<()>> = vec![];
    for i in 0..n_devices {
        let total_throughput_clone = Arc::clone(&total_throughput);
        let c = barrier.clone();
        let devices = devices.clone();
        let mut slices = slices.clone();
        let handle = tokio::spawn(async move {
            let args = env::args().collect::<Vec<_>>();
            
            let id = if party_id == 0 {
                COMM_ID[i]
            } else {
                let res = reqwest::get(format!("http://{}/{}", args[2], i)).await.unwrap();
                IdWrapper::from_str(&res.text().await.unwrap()).unwrap().0
            };
            
            println!("starting device {i}...");
            c.wait().await;

            let comm = Comm::from_rank(devices[i].clone(), party_id, 2, id).unwrap();
            
            let peer_party: i32 = (party_id as i32 + 1) % 2;

            if party_id == 0 {
                println!("sending from {} to {}....", party_id + i, peer_party);
                comm.send(&slices[i], peer_party).unwrap();
                devices[i].synchronize();
                println!("sent from {} to {}!", party_id + i, peer_party);
            } else {
                let now = Instant::now();
                comm.recv(&mut slices[i], peer_party).unwrap();
                devices[i].synchronize();
                let elapsed = now.elapsed();
                let throughput =
                    (DUMMY_DATA_LEN as f64) / (elapsed.as_millis() as f64) / 1_000_000_000f64
                        * 1_000f64;
                println!(
                    "received in {:?} [{:.2} GB/s] [{:.2} Gbps]",
                    elapsed,
                    throughput,
                    throughput * 8f64
                );

                total_throughput_clone.fetch_add(throughput * 8f64, SeqCst);
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.await?;
    }

    println!("Total throughput: {:.2} Gbps", total_throughput.load(Acquire));

    Ok(())
}
