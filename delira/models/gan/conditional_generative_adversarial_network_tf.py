import logging
import tensorflow as tf
import typing
import numpy as np

from delira.models.abstract_network import AbstractTfNetwork




from delira.models.gan.cgan_models import Discriminator
from delira.models.gan.cgan_models import Generator


tf.keras.backend.set_image_data_format('channels_first')


logger = logging.getLogger(__name__)

class ConditionalGenerativeAdversarialNetworkBaseTf(AbstractTfNetwork):
    """Implementation of Conditional GAN  based on pix2pix architecture to create 256x256 pixel or higher resolution images

        Notes
        -----


        References
        ----------
        https://phillipi.github.io/pix2pix/
        https://github.com/yenchenlin/pix2pix-tensorflow

        See Also
        --------
        :class:`AbstractTfNetwork`

    """
    def __init__(self, image_size: int, input_c_dim: int, output_c_dim: int, df_dim: int,
                 gf_dim: int, f_size: int, strides: int, l1_lambda: int, **kwargs):
        """

        Constructs graph containing model definition and forward pass behavior

        """
        super().__init__(image_size=image_size, input_c_dim=input_c_dim,
                         output_c_dim=output_c_dim, df_dim=df_dim, gf_dim=gf_dim, f_size=f_size,
                         strides=strides, l1_lambda=l1_lambda,  **kwargs)


        self.l1_lambda = l1_lambda

        real_data = tf.placeholder(shape=[None, input_c_dim + output_c_dim, image_size, image_size], dtype=tf.float32)
        real_A = real_data[:, :input_c_dim, :, :]
        self.real_B = real_data[:, input_c_dim:input_c_dim + output_c_dim, :, :]
        real_AB = tf.concat([real_A, self.real_B], 1)

        discr, gen = self._build_models(output_c_dim, df_dim, gf_dim, f_size, strides)

        self.gen = gen
        self.discr = discr

        self.fake_B = self.gen(real_A)
        fake_AB = tf.concat([real_A, self.fake_B], 1)

        discr_real = self.discr(real_AB)
        discr_fake = self.discr(fake_AB)

        self.inputs = [real_data]
        self.outputs_train = [self.fake_B, discr_real, discr_fake]
        self.outputs_eval = [self.fake_B, discr_real, discr_fake]

        for key, value in kwargs.items():
            setattr(self, key, value)

    def _add_losses(self, losses: dict):
        """
        Adds losses to model that are to be used by optimizers or during evaluation

        Parameters
        ----------
        losses : dict
            dictionary containing all losses. Individual losses are averaged for discr_real, discr_fake and gen
        """
        if self._losses is not None and len(losses) != 0:
            logging.warning('Change of losses is not yet supported')
            raise NotImplementedError()
        elif self._losses is not None and len(losses) == 0:
            pass
        else:
            self._losses = {}

            total_loss_discr_fake = []
            total_loss_discr_real = []
            total_loss_gen = []

            for name, _loss in losses.items():
                loss_val = tf.reduce_mean(_loss(multi_class_labels=tf.ones_like(self.outputs_train[1]), logits=self.outputs_train[1]))
                self._losses[name + 'discr_real'] = loss_val
                total_loss_discr_real.append(loss_val)

            for name, _loss in losses.items():
                loss_val = tf.reduce_mean(_loss(multi_class_labels=tf.zeros_like(self.outputs_train[2]), logits=self.outputs_train[2]))
                self._losses[name + 'discr_fake'] = loss_val
                total_loss_discr_fake.append(loss_val)

            for name, _loss in losses.items():
                loss_val = tf.reduce_mean(_loss(multi_class_labels=tf.ones_like(self.outputs_train[2]), logits=self.outputs_train[2]))\
                                           + self.l1_lambda * tf.reduce_mean(tf.abs(self.real_B - self.outputs_train[0]))
                self._losses[name + 'gen'] = loss_val
                total_loss_gen.append(loss_val)

            total_loss_discr = tf.reduce_mean([*total_loss_discr_real, *total_loss_discr_fake], axis=0)
            self._losses['total_discr'] = total_loss_discr

            total_loss_gen = tf.reduce_mean(total_loss_gen)
            self._losses['total_gen'] = total_loss_gen

            self.outputs_train.append(self._losses)
            self.outputs_eval.append(self._losses)


    def _add_optims(self, optims: dict):
        """
        Adds optims to model that are to be used by optimizers or during training

        Parameters
        ----------
        optim: dict
            dictionary containing all optimizers, optimizers should be of Type[tf.train.Optimizer]
        """
        if self._optims is not None and len(optims) != 0:
            logging.warning('Change of optims is not yet supported')
            pass
        elif self._optims is not None and len(optims) == 0:
            pass
        else:
            self._optims = optims

            optim_gen = self._optims['gen']
            grads_gen = optim_gen.compute_gradients(self._losses['total_gen'], var_list=self.gen.trainable_variables)
            step_gen = optim_gen.apply_gradients(grads_gen)

            optim_discr = self._optims['discr']
            grads_discr = optim_discr.compute_gradients(self._losses['total_discr'], var_list=self.discr.trainable_variables)
            step_discr = optim_discr.apply_gradients(grads_discr)

            steps = tf.group([step_gen, step_discr])

            self.outputs_train.append(steps)


    @staticmethod
    def _build_models(output_c_dim, df_dim, gf_dim, f_size, strides):
        """
        builds generator and discriminators

        Parameters
        ----------

        Returns
        -------
        tf.keras.Sequential
            created gen
        tf.keras.Sequential
            created discr
        """


        discr = Discriminator(df_dim, f_size)
        gen = Generator(output_c_dim, gf_dim, f_size)

        #discr = Discriminator(df_dim, f_size)
        #gen = Generator(output_c_dim, gf_dim, f_size)

        return discr, gen




    @staticmethod
    def closure(model: typing.Type[AbstractTfNetwork], data_dict: dict,
                metrics={}, fold=0, **kwargs):
        """
                closure method to do a single prediction.
                This is followed by backpropagation or not based state of
                on model.train

                Parameters
                ----------
                model: AbstractTfNetwork
                    AbstractTfNetwork or its child-clases
                data_dict : dict
                    dictionary containing the data
                metrics : dict
                    dict holding the metrics to calculate
                fold : int
                    Current Fold in Crossvalidation (default: 0)
                **kwargs:
                    additional keyword arguments

                Returns
                -------
                dict
                    Metric values (with same keys as input dict metrics)
                dict
                    Loss values (with same keys as those initially passed to model.init).
                    Additionally, a total_loss key is added
                list
                    Arbitrary number of predictions as np.array

                """

        loss_vals = {}
        metric_vals = {}
        image_name_real_fl = "real_images_frontlight"
        image_name_real_bl = "real_images_backlight"
        image_name_fake_fl = "fake_images_frontlight"
        image_name_fake_bl = "fake_images_backlight"
        mask = "segmentation mask"


        input_B = data_dict.pop('data')
        input_A = data_dict.pop('seg')

        inputs = np.concatenate((input_A, input_B), axis=1)

        fake_images, discr_real, discr_fake, losses, *_ = model.run(inputs)

        fake_fl = fake_images[:, :3, :, :]
        fake_bl = fake_images[:, 3:, :, :]
        fake_bl = np.concatenate([fake_bl, fake_bl, fake_bl], 1)
        real_fl = input_B[:, :3, :, :]
        real_bl = input_B[:, 3:, :, :]
        real_bl = np.concatenate([real_bl, real_bl, real_bl], 1)

        for key, loss_val in losses.items():
            loss_vals[key] = loss_val


        if model.training == False:
            # add prefix "val" in validation mode
            eval_loss_vals, eval_metrics_vals = {}, {}
            for key in loss_vals.keys():
                eval_loss_vals["val_" + str(key)] = loss_vals[key]

            for key, metric_fn in metrics.items():
                metric_fl = metric_fn(real_fl, fake_fl)
                metric_vals[key + '_FL_mean'] = np.mean(metric_fl)
                metric_bl = metric_fn(real_bl, fake_bl)
                metric_vals[key + '_BL_mean'] = np.mean(metric_bl)

            for key in metric_vals:
                eval_metrics_vals["val_" + str(key)] = metric_vals[key]

            loss_vals = eval_loss_vals
            metric_vals = eval_metrics_vals

            image_name_real_fl = "val_" + str(image_name_real_fl)
            image_name_real_bl = "val_" + str(image_name_real_bl)
            image_name_fake_fl = "val_" + str(image_name_fake_fl)
            image_name_fake_bl = "val_" + str(image_name_fake_bl)
            mask = "val_" + str(mask)



        for key, val in {**metric_vals, **loss_vals}.items():
            logging.info({"value": {"value": val.item(), "name": key,
                                    "env_appendix": "_%02d" % fold
                                    }})
        hist_name = "Fake_images"

        logging.info({"histogram": {"array": fake_images, "name": hist_name,
                                  "title": "output_image", "env_appendix": "_%02d" % fold}})

        fake_fl = (fake_fl + 1)/2
        logging.info({"images": {"images": fake_fl, "name": image_name_fake_fl,
                                     "title": "output_image", "env_appendix": "_%02d" % fold}})

        fake_bl = (fake_bl + 1)/2


        logging.info({"images": {"images": fake_bl, "name": image_name_fake_bl,
                                     "title": "output_image", "env_appendix": "_%02d" % fold}})
        gt = input_A

        logging.info({"images": {"images": np.concatenate([gt, gt, gt], 1), "name": mask,
                                     "title": "input_image", "env_appendix": "_%02d" % fold}})

        real_fl = (real_fl + 1)/2

        logging.info({"images": {"images": real_fl, "name": image_name_real_fl,
                                     "title": "input_image", "env_appendix": "_%02d" % fold}})

        real_bl = (real_bl + 1)/2



        logging.info({"images": {"images": real_bl, "name": image_name_real_bl,
                                     "title": "input_image", "env_appendix": "_%02d" % fold}})

        return metric_vals, loss_vals, [fake_images, discr_fake, discr_real]


