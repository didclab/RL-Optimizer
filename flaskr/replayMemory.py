import numpy as np
from multiprocessing import shared_memory

from config import *

N_STATE_VAR = 8

class ReplayMemory(object):
    def __init__(self, n_state_var=N_STATE_VAR):
        self.mem_count = 0
        self.sample_count = 1
        self.refresh_freq = 50
        
        states_ = np.zeros((REPLAYMEM_SIZE, n_state_var),dtype=np.float64)
        self.shm_states = shared_memory.SharedMemory(create=True, size=states_.nbytes, name='states')
        self.states = np.ndarray(states_.shape, dtype=states_.dtype, buffer=self.shm_states.buf)
        
        actions_ = np.zeros(REPLAYMEM_SIZE, dtype=np.int64)
        self.shm_actions = shared_memory.SharedMemory(create=True, size=actions_.nbytes, name='actions')
        self.actions = np.ndarray(actions_.shape, dtype=actions_.dtype, buffer=self.shm_actions.buf)
        
        rewards_ = np.zeros(REPLAYMEM_SIZE, dtype=np.float64)
        self.shm_rewards = shared_memory.SharedMemory(create=True, size=rewards_.nbytes, name='rewards')
        self.rewards = np.ndarray(rewards_.shape, dtype=rewards_.dtype, buffer=self.shm_rewards.buf)
        
        nstates_ = np.zeros((REPLAYMEM_SIZE, n_state_var),dtype=np.float64)
        self.shm_nstates = shared_memory.SharedMemory(create=True, size=nstates_.nbytes, name='nstates')
        self.nstates = np.ndarray(nstates_.shape, dtype=nstates_.dtype, buffer=self.shm_nstates.buf)
        
        dones_ = np.zeros(REPLAYMEM_SIZE, dtype=bool)
        self.shm_dones = shared_memory.SharedMemory(create=True, size=dones_.nbytes, name='dones')
        self.dones = np.ndarray(dones_.shape, dtype=dones_.dtype, buffer=self.shm_dones.buf)
        
        logp_ = np.zeros(REPLAYMEM_SIZE, dtype=np.float64)
        self.shm_logp = shared_memory.SharedMemory(create=True, size=logp_.nbytes, name='logp')
        self.logp = np.ndarray(logp_.shape, dtype=logp_.dtype, buffer=self.shm_logp.buf)
        
        I_ = np.zeros(REPLAYMEM_SIZE, dtype=np.float64)
        self.shm_I = shared_memory.SharedMemory(create=True, size=I_.nbytes, name='I')
        self.I = np.ndarray(I_.shape, dtype=I_.dtype, buffer=self.shm_I.buf)
        
        error_ = np.zeros(REPLAYMEM_SIZE, dtype=np.float64)
        self.shm_error = shared_memory.SharedMemory(create=True, size=error_.nbytes, name='error')
        self.error = np.ndarray(error_.shape, dtype=error_.dtype, buffer=self.shm_error.buf)
    
    def free(self):
        self.shm_states.close()
        self.shm_states.unlink()
        
        self.shm_actions.close()
        self.shm_actions.unlink()
        
        self.shm_rewards.close()
        self.shm_rewards.unlink()
        
        self.shm_nstates.close()
        self.shm_nstates.unlink()
        
        self.shm_dones.close()
        self.shm_dones.unlink()
        
        self.shm_logp.close()
        self.shm_logp.unlink()
        
        self.shm_I.close()
        self.shm_I.unlink()
        
        self.shm_error.close()
        self.shm_error.unlink()
    
    def __len__(self):
        return min(REPLAYMEM_SIZE, self.mem_count)
    
    def g_add(self, state, action, reward, nstate, done, logp, I, error=0.):
        mem_index = self.mem_count % REPLAYMEM_SIZE
        
        self.states[mem_index]  = state
        self.actions[mem_index] = action
        self.rewards[mem_index] = reward
        self.nstates[mem_index] = nstate
        self.dones[mem_index] =  1 - done
        self.logp[mem_index] = -logp
        self.I[mem_index] = I
        self.error[mem_index] = error

        self.mem_count += 1
    
    def priority_sample(self):
        REPMEM_MAX = min(self.mem_count, REPLAYMEM_SIZE)
        batch_indices = self.error[:REPMEM_MAX].argsort()[::-1][:MINIBAT_SIZE]

        states  = self.states[batch_indices]
        actions = self.actions[batch_indices]
        rewards = self.rewards[batch_indices]
        nstates = self.nstates[batch_indices]
        dones   = self.dones[batch_indices]
        logp    = self.logp[batch_indices]
        I       = self.I[batch_indices]
        
        if self.sample_count % self.refresh_freq == 0:
            self.error[batch_indices][:16] = 0.
        self.sample_count += 1
        
        return states, actions, rewards, nstates, dones, logp, I
        
    
    def random_sample(self):
        REPMEM_MAX = min(self.mem_count, REPLAYMEM_SIZE)
        batch_indices = np.random.choice(REPMEM_MAX, MINIBAT_SIZE, replace=True)
        
        states  = self.states[batch_indices]
        actions = self.actions[batch_indices]
        rewards = self.rewards[batch_indices]
        nstates = self.nstates[batch_indices]
        dones   = self.dones[batch_indices]
        logp    = self.logp[batch_indices]
        I       = self.I[batch_indices]
        self.sample_count += 1

        return states, actions, rewards, nstates, dones, logp, I

class ReplaySlice(object):
    def __init__(self, N, n_state_var=N_STATE_VAR):
        self.mem_count = REPLAYMEM_WORKER_SIZE
        self.frm = N * REPLAYMEM_WORKER_SIZE
        
        self.shm_states = shared_memory.SharedMemory(name='states')
        self.states = np.ndarray((REPLAYMEM_SIZE, n_state_var), dtype=np.float64, buffer=self.shm_states.buf)
        
        self.shm_actions = shared_memory.SharedMemory(name='actions')
        self.actions = np.ndarray(REPLAYMEM_SIZE, dtype=np.int64, buffer=self.shm_actions.buf)
        
        self.shm_rewards = shared_memory.SharedMemory(name='rewards')
        self.rewards = np.ndarray(REPLAYMEM_SIZE, dtype=np.float64, buffer=self.shm_rewards.buf)
        
        self.shm_nstates = shared_memory.SharedMemory(name='nstates')
        self.nstates = np.ndarray((REPLAYMEM_SIZE, n_state_var), dtype=np.float64, buffer=self.shm_nstates.buf)
        
        self.shm_dones = shared_memory.SharedMemory(name='dones')
        self.dones = np.ndarray(REPLAYMEM_SIZE, dtype=bool, buffer=self.shm_dones.buf)
        
        self.shm_logp = shared_memory.SharedMemory(name='logp')
        self.logp = np.ndarray(REPLAYMEM_SIZE, dtype=np.float64, buffer=self.shm_logp.buf)
        
        self.shm_I = shared_memory.SharedMemory(name='I')
        self.I = np.ndarray(REPLAYMEM_SIZE, dtype=np.float64, buffer=self.shm_I.buf)
        
        self.shm_error = shared_memory.SharedMemory(name='error')
        self.error = np.ndarray(REPLAYMEM_SIZE, dtype=np.float64, buffer=self.shm_error.buf)
        
    def free(self):
        self.shm_states.close()
        self.shm_actions.close()
        self.shm_rewards.close()
        self.shm_nstates.close()
        self.shm_dones.close()
        self.shm_logp.close()
        self.shm_I.close()
        self.shm_error.close()
        
    def add(self, state, action, reward, nstate, done, logp, I, error=0.):
        mem_index = self.frm + (self.mem_count % REPLAYMEM_WORKER_SIZE)
        
        self.states[mem_index]  = state
        self.actions[mem_index] = action
        self.rewards[mem_index] = reward
        self.nstates[mem_index] = nstate
        self.dones[mem_index] =  1 - done
        self.logp[mem_index] = -logp
        self.I[mem_index] = I
        self.error[mem_index] = error

        self.mem_count += 1
        
    def __len__(self):
        return min(REPLAYMEM_SIZE, self.mem_count)

