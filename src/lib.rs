use cudarc::{cublas::CudaBlas, driver::{CudaDevice, CudaSlice}};

pub trait GPUDotEngine {
    fn create_database(&mut self, db_entries: &[u16], element_size: usize);
    fn prepare_query(&self, query: &[u16]) -> &[u16];
    fn dot(&self, query: &[u16]) -> &[u16];
}

pub struct U16RingEngine {
    width: usize,
    length: usize,
    db1: CudaSlice<u8>,
    db0: CudaSlice<u8>,
    db1_sums: CudaSlice<u8>,
    db0_sums: CudaSlice<u8>,
}

impl GPUDotEngine for U16RingEngine {
    fn create_database(&mut self, db_entries: &[u16], element_size: usize) {
        self.width = element_size;
        self.length = db_entries.len() / self.width;
        let dev = CudaDevice::new(0).unwrap();
        let blas = CudaBlas::new(dev.clone()).unwrap();

        let mut a1_host = db_entries.iter().map(|x| (x >> 8) as u8).collect::<Vec<_>>();
        let a1_sums: Vec<u16> = a1_host.chunks(element_size)
            .map(|row| row.iter().map(|&x| x as u16).sum())
            .collect();
        a1_host.iter_mut().for_each(|x| (*x = (*x as i8 - 127 - 1) as u8));

        let mut a0_host = db_entries.iter().map(|x| (x >> 8) as u8).collect::<Vec<_>>();
        let a0_sums: Vec<u16> = a0_host.chunks(element_size)
            .map(|row| row.iter().map(|&x| x as u16).sum())
            .collect();
        a0_host.iter_mut().for_each(|x| (*x = (*x as i8 - 127 - 1) as u8));

        let a1_dev = dev.htod_sync_copy(&a1_host).unwrap();
        let a0_dev = dev.htod_sync_copy(&a0_host).unwrap();
        let a1_sums_dev = dev.htod_sync_copy(&a1_sums).unwrap();
        let a0_sums_dev = dev.htod_sync_copy(&a0_sums).unwrap();
    }

    fn prepare_query(&self, query: &[u16]) -> &[u16] {
        todo!()
    }

    fn dot(&self, query: &[u16]) -> &[u16] {
        todo!()
    }
}
