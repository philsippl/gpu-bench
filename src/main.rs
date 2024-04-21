use std::{env, str::FromStr, time::Instant};

use cudarc::{
    driver::{CudaDevice, CudaSlice},
    nccl::{
        group_end, group_start,
        result::{all_reduce, comm_init_rank, get_uniqueid},
        Comm, Id, ReduceOp,
    },
};

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

// 1 GB
const LEN: usize = 20 * (1 << 30);

fn main() {
    // NCCL_COMM_ID
    // let n_devices = CudaDevice::count().unwrap() as usize;

    let args = env::args().collect::<Vec<_>>();
    let rank = args[1].parse().unwrap();
    let device_id = args[2].parse().unwrap();
    let n_devices = args[3].parse().unwrap();

    println!("000000");

    let id = if rank == 0 {
        let id = Id::new().unwrap();
        println!("{:?}", IdWrapper(id.clone()).to_string());
        id
    } else {
        let id = IdWrapper::from_str(&args[4]).unwrap().0;
        println!("{:?}", IdWrapper(id.clone()).to_string());
        id
    };

    println!("1111");

    // hex::encode(id.internal());
    let dev = CudaDevice::new(device_id).unwrap();
    let comm = Comm::from_rank(dev.clone(), rank, n_devices, id).unwrap();

    println!("222");

    // let slice = dev.htod_copy(vec![1337 as i32]).unwrap();
    
    let peer: i32 = (rank as i32 + 1) % 2;
    
    if rank == 0 {
        let slice: CudaSlice<u8> = dev.alloc_zeros(LEN).unwrap();
        println!("sending from {} to {}: {:?}", rank, peer, slice);
        comm.send(&slice, peer).unwrap();
        println!("sent from {} to {}: {:?}", rank, peer, slice);
    } else {
        let mut slice_receive = dev.alloc_zeros::<u8>(LEN).unwrap();
        println!("waiting for msg from peer {} ...", peer);
        let now = Instant::now();
        comm.recv(&mut slice_receive, peer).unwrap();
        dev.synchronize().unwrap();
        let elapsed = now.elapsed();
        println!(
            "received in {:?} ({:.2} GB/s)",
            elapsed,
            (LEN as f64) / (elapsed.as_millis() as f64) / 1_000_000_000f64 * 1_000f64 * 8f64
        );
        // let out = dev.dtoh_sync_copy(&slice_receive).unwrap();
        // println!("GPU {} received from peer {}: {:?}", rank, peer, out);
    }

    std::thread::sleep(std::time::Duration::from_secs(30));
}
