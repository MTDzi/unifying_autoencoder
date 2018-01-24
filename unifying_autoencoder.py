from keras.layers import Input, Dense, Dropout, ELU
from keras.models import Model
from keras.utils.np_utils import to_categorical
from keras.regularizers import l2


class UnifyingAutoEncoder:
    def __init__(self, inp_shapes,
                 unifying_dim=100, num_unifying_layers=2,
                 num_neurons=40, num_layers=3,
                 num_epochs=10,reg=1e-3):
        self.inp_shapes = inp_shapes
        self.unifying_dim = unifying_dim
        self.num_unifying_layers = 1
        self.num_neurons = num_neurons
        self.num_layers = num_layers
        self.reg = reg
        self.num_epochs = num_epochs
        self.num_epochs_so_far = 0
        self._init_model()

    def fit(self, Xs, verbose=1):
        self.num_epochs_so_far += self.num_epochs
        self.re_creation_model.fit(
            x=Xs,
            y=Xs,
            verbose=verbose,
            epochs=self.num_epochs,
        )
        return self

    def unify(self, Xs):
        return self.unification_model.predict(Xs)

    def _init_model(self):
        # We initialize as many `Input`s as we had elements in `self.inp_shapes`
        inputs = [Input((shape, )) for shape in self.inp_shapes]

        # Here, we are creating layers which unify the dimensionality
        unifying_strata = [
            Dense(self.unifying_dim)(inp) for inp in inputs
        ]
        unifying_strata = [
            ELU()(stratum) for stratum in unifying_strata
        ]
        for _ in range(self.num_unifying_layers-1):
            unifying_strata = [Dense(self.unifying_dim)(inp) for stratum in unifying_strata]
            unifying_strata = [ELU()(stratum) for stratum in unifying_strata]

        # Once the dimensionality is unified, we can apply the AE
        base_network = self._create_base_network()
        ae_strata = [
            base_network(stratum) for stratum in unifying_strata
        ]

        # Now, once we got output from the `base_network`, we can re-create our input
        unifying_strata_after_ae = [
            Dense(self.unifying_dim)(stratum) for stratum in ae_strata
        ]
        unifying_strata_after_ae = [
            ELU()(stratum) for stratum in unifying_strata_after_ae
        ]

        # We need to expand the dimensions to those that we had on input
        output_strata = [
            Dense(out_shape)(uni_stratum)
            for uni_stratum, out_shape in zip(unifying_strata_after_ae, self.inp_shapes)
        ]

        # We'll need this to re-create the input (and check if it makes sense)
        self.re_creation_model = Model(inputs=inputs, outputs=output_strata)
        self.re_creation_model.compile(
            optimizer='adam',
            loss='mean_squared_error',
        )

        # And now for the unification model
        self.unification_model = Model(
            inputs=inputs,
            outputs=unifying_strata_after_ae,
        )

        # central_layer_id = 1 + self.num_layers // 2
        # central_ae_layer = base_network.layers[central_layer_id]
        # central_strata = [central_ae_layer(inp) for inp in inputs]
        # self.unification_model = Model(
        #     inputs=inputs,
        #     outputs=central_strata,
        # )

    def _create_base_network(self):
        inp = Input(shape=(self.unifying_dim, ))
        x = Dense(self.num_neurons, kernel_regularizer=l2(self.reg))(inp)
        x = ELU()(x)
        for _ in range(self.num_layers-1):
            x = Dense(self.num_neurons, kernel_regularizer=l2(self.reg))(x)
            x = ELU()(x)
        return Model(inputs=inp, outputs=x)
