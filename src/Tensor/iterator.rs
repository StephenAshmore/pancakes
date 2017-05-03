use Tensor::Vector;
use rand;
use rand::distributions::{Range, IndependentSample};

pub struct Iterator {
    m_size: u64,
    m_random: rand::ThreadRng,
    m_range: Range<u64>,
    m_list: Vector<u64>,
    m_next: u64,
}

impl Iterator {
    pub fn new(size: u64) -> Iterator
    {
        let mut new_list:Vector<u64> = Vector::with_capacity(size);
        new_list.resize(size, 0);

        Iterator
        {
            m_size: size,
            m_random: rand::thread_rng(),
            m_range: Range::new(0, size),
            m_list: new_list,
            m_next: 0,
        }
    }

    pub fn has_next(&self) -> bool
    {
        if self.m_next >= self.m_list.size() {
            return false
        }
        true
    }

    pub fn next(&mut self) -> u64
    {
        if self.m_next >= self.m_list.size()
        {
            self.reset()
        }
        self.m_next += 1;
        self.m_list[self.m_next - 1]
    }

    pub fn reset(&mut self)
    {
        self.m_next = 0;
        // reset list
        for i in 0..self.m_size
        {
            self.m_list[i] = i;
        }

        // shuffle list with fisher-yates/durstenfeld shuffle:
        let mut temp:u64; let mut index:u64;
        for i in (1u64..self.m_size-1).rev()
        {
            index = self.m_range.ind_sample(&mut self.m_random) % i;
            temp = self.m_list[i];
            self.m_list[i] = self.m_list[index];
            self.m_list[index] = temp;
        }
    }
}