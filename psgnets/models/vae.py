import tensorflow.compat.v1 as tf
from vvn.ops.convolutional import mlp

PRINT = False

# standard VAE
class VAE(object):

    def __init__(self,
                 encoder_dims=[50],
                 latent_dims=5,
                 decoder_dims=[50],
                 output_dims=None,
                 activations=tf.nn.relu,
                 beta=10.0,
                 log_sigma_max=5.,
                 kernel_initializer=tf.variance_scaling_initializer(seed=0, scale=0.001), **kwargs
    ):
        self.encoder_dims = encoder_dims
        self.latent_dims = latent_dims
        self.decoder_dims = decoder_dims
        self.activations = activations
        self.kernel_initializer = kernel_initializer
        self.beta = beta
        self.log_sigma_max = log_sigma_max
        self.D = output_dims

    def _encoder(self, x):
        if not self.D:
            self.D = x.shape.as_list()[-1]
        assert x.shape.as_list()[-1] == self.D
        x = mlp(x,
                hidden_dims=self.encoder_dims,
                activations=[self.activations]*len(self.encoder_dims),
                kernel_initializer=self.kernel_initializer,
                scope="vae_encoder")
        mu = mlp(x, hidden_dims=self.latent_dims, activations=tf.identity, kernel_initializer=self.kernel_initializer, scope="mu")
        log_sigma = mlp(x, hidden_dims=self.latent_dims, activations=tf.identity, kernel_initializer=tf.constant_initializer(0.0), scope="log_sigma")
        return mu, log_sigma

    def _sample_z(self, mu, log_sigma):
        eps = tf.random_normal(shape=tf.shape(mu), seed=0, name="eps") # [?,L]
        # log_sigma = tf.minimum(log_sigma, self.log_sigma_max)
        z = mu + eps*tf.exp(log_sigma) # [?,L]
        return z

    def _decoder(self, z):
        y = mlp(z,
                hidden_dims=(self.decoder_dims + [self.D]),
                kernel_initializer=self.kernel_initializer,
                activations=([self.activations]*(len(self.decoder_dims)) + [lambda t: tf.identity(t, name='vae_output')]), scope="vae_decoder")

        return y

    def _vae_loss(self, x, x_recon, mu, log_sigma, beta):
        recon_loss = tf.reduce_sum(tf.square(x - x_recon), axis=-1) # [?]
        # log_sigma = tf.minimum(log_sigma, self.log_sigma_max)
        kl_loss = tf.reduce_sum(
            -0.5 * (1. + (2.*log_sigma) - tf.square(mu) - tf.exp(2.*log_sigma)),
            axis=-1) # [?]
        if PRINT:
            kl_loss = tf.Print(kl_loss, [tf.reduce_mean(kl_loss), tf.reduce_mean(recon_loss), tf.reduce_max(mu), tf.reduce_max(log_sigma)], message='kl_loss/recon_loss/mu/lsig')
        vae_loss = tf.reduce_mean(recon_loss + beta * kl_loss)
        return tf.reshape(vae_loss, [1])

    def predict(self, x):

        mu, log_sigma = self._encoder(x)
        z = self._sample_z(mu, log_sigma)
        x_recon = self._decoder(z)
        loss = self._vae_loss(x, x_recon, mu, log_sigma, beta=self.beta)
        return x_recon, loss
