// node 1: FI_EFA_USE_DEVICE_RDMA=1 NCCL_NET=socket ./target/release/gpu-bench 0 0 2
// node 2: FI_EFA_USE_DEVICE_RDMA=1 NCCL_NET=socket NCCL_SOCKET_IFNAME=ens32 NCCL_DEBUG=INFO ./target/release/gpu-bench 1 0 2 ea4ce654861caacd0200c787ac1f29530000000000000000000000000000000000000000000000001019794c85550000b0e20ce2ff7f00000050b2c8b87f0000b0e00ce2ff7f000082f38fc8b87f00000100000000000000100000003000000080e20ce2ff7f0000b0e10ce2ff7f00000180adfb855500002919794c85550000

use std::{
    env,
    str::FromStr,
    time::{Duration, Instant},
};

use axum::{extract::Path, routing::get, Router};
use cudarc::{
    driver::{CudaDevice, CudaSlice},
    nccl::{
        Comm, Id,
    },
};
use once_cell::sync::Lazy;
use tokio::time::sleep;

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
    tracing_subscriber::fmt::init();
    let args = env::args().collect::<Vec<_>>();
    let n_devices = CudaDevice::count().unwrap() as usize;
    let party_id: usize = args[1].parse().unwrap();

    for i in 0..n_devices {
        tokio::spawn(async move {
            let args = env::args().collect::<Vec<_>>();

            let id = if party_id == 0 {
                COMM_ID[i]
            } else {
                let res = reqwest::get(format!("http://{}/{}", args[2], i)).await.unwrap();
                IdWrapper::from_str(&res.text().await.unwrap()).unwrap().0
            };

            let dev = CudaDevice::new(i).unwrap();
            let mut slice: CudaSlice<u8> = dev.alloc_zeros(DUMMY_DATA_LEN).unwrap();
            let comm = Comm::from_rank(dev.clone(), party_id, n_devices, id).unwrap();

            let peer_party: i32 = (party_id as i32 + 1) % 2;

            if party_id == 0 {
                println!("sending from {} to {}....", party_id + i, peer_party);
                comm.send(&slice, peer_party).unwrap();
            } else {
                let now = Instant::now();
                comm.recv(&mut slice, peer_party).unwrap();
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
            }
        });
    }

    if party_id == 0 {
        tokio::spawn(async move {
            let app = Router::new().route("/:device_id", get(root));

            let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
            axum::serve(listener, app).await.unwrap();
        }).await?;
    };

    Ok(())
}

// fn main() {
//     // NCCL_COMM_ID
//     // let n_devices = CudaDevice::count().unwrap() as usize;

//     let args = env::args().collect::<Vec<_>>();
//     let rank = args[1].parse().unwrap();
//     let device_id = args[2].parse().unwrap();
//     let n_devices = args[3].parse().unwrap();

//     println!("000000");

//     let id = if rank == 0 {
//         let id = Id::new().unwrap();
//         println!("{:?}", IdWrapper(id.clone()).to_string());
//         id
//     } else {
//         let id = IdWrapper::from_str(&args[4]).unwrap().0;
//         println!("{:?}", IdWrapper(id.clone()).to_string());
//         id
//     };

//     println!("1111");

//     // hex::encode(id.internal());
//     let dev = CudaDevice::new(device_id).unwrap();

//     let mut slice: CudaSlice<u8> = dev.alloc_zeros(LEN).unwrap();

//     let comm = Comm::from_rank(dev.clone(), rank, n_devices, id).unwrap();

//     println!("222");

//     // let slice = dev.htod_copy(vec![1337 as i32]).unwrap();

//     let peer: i32 = (rank as i32 + 1) % 2;

//     for i in 0..10 {
//         if rank == 0 {
//             println!("sending from {} to {}: {:?}", rank, peer, slice);
//             comm.send(&slice, peer).unwrap();
//             println!("sent from {} to {}: {:?}", rank, peer, slice);
//         } else {
//             println!("waiting for msg from peer {} ...", peer);
//             let now = Instant::now();
//             comm.recv(&mut slice, peer).unwrap();
//             dev.synchronize().unwrap();
//             let elapsed = now.elapsed();
//             let throughput =(LEN as f64) / (elapsed.as_millis() as f64) / 1_000_000_000f64 * 1_000f64;
//             println!(
//                 "received in {:?} [{:.2} GB/s] [{:.2} Gbps]",
//                 elapsed,
//                 throughput,
//                 throughput * 8f64
//             );
//         }
//     }

//     std::thread::sleep(std::time::Duration::from_secs(30));
// }
