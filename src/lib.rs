#[macro_use]
extern crate lazy_static;

pub mod alphazero;
pub mod backgammon;
pub mod mcts;
pub mod versus;

/*

Constants used through all modules
*/
pub mod constants {
    use tch::Device;

    lazy_static! {
        pub static ref DEVICE: Device = {
            if tch::utils::has_mps() {
                Device::Mps
            } else {
                Device::cuda_if_available()
            }
        };
    }
}
