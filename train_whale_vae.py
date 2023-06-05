from deepscm_vae import whalecalls
import torch

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_epochs = 500
    save_images_every = 1
    l_rate = 1e-4
    batch_size = 32
    image_output_path = "vae_whale_results"
    vae, _ = whalecalls.train("WhaleData/Nocall", "WhaleData/Gunshot", "WhaleData/Upcall",
                              n_epochs=n_epochs, device=device, save_images_every=save_images_every,
                              image_output_path=image_output_path, batch_size=batch_size, l_rate=l_rate)
    torch.save({
        "vae": vae
    }, "whale_vae.tar")
