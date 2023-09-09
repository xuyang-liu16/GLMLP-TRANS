import tensorflow as tf

# Window_size
W = 500
# Feature_num (Acc、Gyr、Mag、LAcc、Gra、Ori、Pressure)
F = 10

# Imput Layer
input_LAcc_x = tf.keras.layers.Input(shape=(W, 1))
input_LAcc_y = tf.keras.layers.Input(shape=(W, 1))
input_LAcc_z = tf.keras.layers.Input(shape=(W, 1))

input_Gyr_x = tf.keras.layers.Input(shape=(W, 1))
input_Gyr_y = tf.keras.layers.Input(shape=(W, 1))
input_Gyr_z = tf.keras.layers.Input(shape=(W, 1))

input_Mag_x = tf.keras.layers.Input(shape=(W, 1))
input_Mag_y = tf.keras.layers.Input(shape=(W, 1))
input_Mag_z = tf.keras.layers.Input(shape=(W, 1))

input_Pressure = tf.keras.layers.Input(shape=(W, 1))



# Merge
input_concat = tf.keras.layers.Concatenate(axis=2)


# GFE Block
# Hyperparameters in GFE Block
hidden_dim = 64
num_heads = 8
key_dim = 64


def GFE_Block(x):
    x = tf.keras.layers.Dense(hidden_dim)(x)  # Project Layer
    x = tf.keras.layers.MultiHeadAttention(num_heads, key_dim)(x, x, x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    return x


# LFE Block
# Hyperparameters in LFE Block
hidden_dim = 64
num_patches = 500
dropout_rate = 0.5
token_mlp_dim = 128  # Ds
channel_mlp_dim = 256  # Dc
num_mixer_layers = 8

# Mixer
cross_time_mixer = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(units=token_mlp_dim, activation="gelu"),  # Mixer (F, T)(T, Ds) => (F, Ds)
        tf.keras.layers.Dense(units=num_patches),  # (F, Ds)(Ds, T) => (F, T)
        tf.keras.layers.Dropout(0.5)
    ], name="cross_time_mixer"
)
cross_senor_mixer = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(units=channel_mlp_dim, activation="gelu"),  # Mixer (T, F)(F, Dc) => (T, Dc)
        tf.keras.layers.Dense(units=hidden_dim),  # (T, Dc)(Dc, F) => (T, F)
        tf.keras.layers.Dropout(0.5)
    ],
    name="cross_sensor_mixer"
)

def LFE_Block(inputs):
    inputs = tf.keras.layers.Dense(hidden_dim)(inputs)  # Project Layer
    x = inputs
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Permute([2, 1])(x)
    x = cross_time_mixer(x)
    x = tf.keras.layers.Permute([2, 1])(x)
    x_shortcut_2 = tf.keras.layers.add([x, inputs])
    x = tf.keras.layers.LayerNormalization()(x_shortcut_2)  # skip connection
    x = cross_senor_mixer(x)
    x = tf.keras.layers.add([x, x_shortcut_2])  # skip connection
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    return x


# Fusion Layer
def Fusion_Layer(x1, x2):
    x1 = tf.keras.layers.Reshape(target_shape=(64, 1))(x1)
    x2 = tf.keras.layers.Reshape(target_shape=(64, 1))(x2)
    x = tf.keras.layers.Concatenate(axis=2)([x1, x2])
    x = tf.keras.layers.Dense(1, activation="relu")(x)
    x = tf.keras.layers.Flatten()(x)
    return x


# MLP Layer
def MLP_Layer(inputs):
    x = inputs
    x = tf.keras.layers.Dense(units=64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(units=128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(units=256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(units=64, activation="relu")(x)
    x = tf.keras.layers.add([x, inputs])

    output = tf.keras.layers.Dense(8, activation="softmax")(x)
    return output


# GLMLP-TRANS
x = input_concat(
    [input_LAcc_x, input_LAcc_y, input_LAcc_z, input_Gyr_x, input_Gyr_y, input_Gyr_z, input_Mag_x, input_Mag_y,
     input_Mag_z, input_Pressure])
x1 = GFE_Block(x)
x2 = LFE_Block(x)
x = Fusion_Layer(x1, x2)
output = MLP_Layer(x)

GLMLP_TRANS = tf.keras.Model(inputs=[
    input_LAcc_x, input_LAcc_y, input_LAcc_z,
    input_Gyr_x, input_Gyr_y, input_Gyr_z,
    input_Mag_x, input_Mag_y, input_Mag_z,
    input_Pressure
], outputs=output)

GLMLP_TRANS.summary()

tf.keras.utils.plot_model(GLMLP_TRANS, to_file="model.png", show_shapes=True)
