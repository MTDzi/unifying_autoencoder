from keras.layers import Input, Dense, Dropout, ELU
from keras.models import Model
from keras.utils.np_utils import to_categorical
from keras.regularizers import l2


class UnifyingClassifier:
    def __init__(self, inp_shapes,
                 unifying_dim=100, num_unifying_layers=3,
                 num_neurons=40, num_layers=3,
                 num_epochs=10, reg=1e-3):
        self.inp_shapes = inp_shapes
        self.unifying_dim = unifying_dim
        self.num_unifying_layers = 1
        self.num_neurons = num_neurons
        self.num_layers = num_layers
        self.reg = reg
        self.num_epochs = num_epochs
        self.num_epochs_so_far = 0
        self._init_model()

    def fit(self, Xs, ys, verbose=1):
        self.num_epochs_so_far += self.num_epochs
        self.cls_model.fit(
            x=Xs,
            y=ys,
            verbose=verbose,
            epochs=self.num_epochs,
        )
        return self

    def predict(self, Xs):
        return self.cls_model.predict(Xs)

    def _init_model(self):
        # We initialize as many `Input`s as we had elements in `self.inp_shapes`
        inputs = [Input((shape, )) for shape in self.inp_shapes]

        # Here, we are creating layers which unify the dimensionality
        unifying_strata = [
            Dense(self.unifying_dim, kernel_regularizer=l2(self.reg))(inp)
            for inp in inputs
        ]
        unifying_strata = [
            ELU()(stratum)
            for stratum in unifying_strata
        ]
        for _ in range(self.num_unifying_layers-1):
            unifying_strata = [
                Dense(self.unifying_dim, kernel_regularizer=l2(self.reg))(stratum)
                for stratum in unifying_strata
            ]
            unifying_strata = [
                ELU()(stratum)
                for stratum in unifying_strata
            ]

        # Once the dimensionality is unified, we can apply the AE
        base_network = self._create_base_network()
        ae_strata = [
            base_network(stratum)
            for stratum in unifying_strata
        ]

        # We need to expand the dimensions to those that we had on input
        output_strata = [
            Dense(1, activation='sigmoid')(stratum)
            for stratum in ae_strata
        ]

        # We'll need this to re-create the input (and check if it makes sense)
        self.cls_model = Model(inputs=inputs, outputs=output_strata)
        self.cls_model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
        )

    def _create_base_network(self):
        inp = Input(shape=(self.unifying_dim, ))
        x = Dense(self.num_neurons, kernel_regularizer=l2(self.reg))(inp)
        x = ELU()(x)
        for _ in range(self.num_layers-1):
            x = Dense(self.num_neurons, kernel_regularizer=l2(self.reg))(x)
            x = ELU()(x)
        return Model(inputs=inp, outputs=x)
