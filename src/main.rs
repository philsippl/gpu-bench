use std::{env, mem::MaybeUninit};

use cudarc::{
    driver::CudaDevice,
    nccl::{group_end, group_start, result::{all_reduce, comm_init_rank, get_uniqueid}, Comm, Id, ReduceOp},
};


fn main() {
    let n_devices = CudaDevice::count().unwrap() as usize;
    let mut threads = vec![];
    let id = Id::new().unwrap();

    for i in 0..n_devices {
        let thread = std::thread::spawn(move || {
            let dev = CudaDevice::new(i).unwrap();
            let comm = Comm::from_rank(dev.clone(), i, n_devices, id).unwrap();
            let slice = dev.htod_copy(vec![i as i32]).unwrap();
            let mut slice_receive = dev.alloc_zeros::<i32>(1).unwrap();

            let peer: i32 = (i as i32 + 1) % 2;

            println!("sending from {} to {}: {:?}", i, peer, slice);
            comm.send(&slice, peer).unwrap();
            println!("sent from {} to {}: {:?}", i, peer, slice);
            comm.recv(&mut slice_receive, peer).unwrap();

            let out = dev.dtoh_sync_copy(&slice_receive).unwrap();
            println!("GPU {} received from peer {}: {:?}", i, peer, out);
        });

        threads.push(thread);
    }

    for t in threads {
        t.join().unwrap();
    }

}
