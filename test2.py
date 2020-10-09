De1 = layers.Conv2D(64, 5, 2, 'same')(input)
# De1: (None, 128, 128, 64)
De2 = layers.LeakyReLU(0.2)(De1)
De2 = layers.Conv2D(64, 3, 1, 'same', activation='relu')(De2)
De2 = layers.Conv2D(128, 5, 2, 'same')(De2)  # Downsampling
De2 = layers.BatchNormalization()(De2)
# De2: (None, 64, 64, 128)
De3 = layers.LeakyReLU(0.2)(De2)
De3 = layers.Conv2D(128, 3, 1, 'same', activation='relu')(De3)
De3 = layers.Conv2D(256, 5, 2, 'same')(De3)  # Downsampling
De3 = layers.BatchNormalization()(De3)
# De3: (None, 32, 32, 256)
De4 = layers.LeakyReLU(0.2)(De3)
De4 = layers.Conv2D(256, 3, 1, 'same', activation='relu')(De4)
De4 = layers.Conv2D(512, 5, 2, 'same')(De4)  # Downsampling
De4 = layers.BatchNormalization()(De4)
# De4: (None, 16, 16, 512)
De5 = layers.LeakyReLU(0.2)(De4)
De5 = layers.Conv2D(512, 3, 1, 'same', activation='relu')(De5)
De5 = layers.Conv2D(512, 5, 2, 'same')(De5)  # Downsampling
De5 = layers.BatchNormalization()(De5)
# De5: (None, 8, 8, 512)
De6 = layers.LeakyReLU(0.2)(De5)
De6 = layers.Conv2D(512, 3, 1, 'same', activation='relu')(De6)
De6 = layers.Conv2D(512, 5, 2, 'same')(De6)  # Dowmsampling
De6 = layers.BatchNormalization()(De6)
# De6: (None, 4, 4, 512)
De7 = layers.LeakyReLU(0.2)(De6)
De7 = layers.Conv2D(512, 3, 1, 'same', activation='relu')(De7)
De7 = layers.Conv2D(512, 5, 2, 'same')(De7)  # Downsampling
De7 = layers.BatchNormalization()(De7)
# De7: (None, 2, 2, 512)
De8 = layers.LeakyReLU(0.2)(De7)
De8 = layers.Conv2D(512, 3, 1, 'same', activation='relu')(De8)
De8 = layers.Conv2D(512, 5, 2, 'same')(De8)  # Downsampling
De8 = layers.BatchNormalization()(De8)
# De8: (None, 1, 1, 512)
# Decoder:
Dd1 = layers.Activation('relu')(De8)
Dd1 = layers.Conv2D(512, 3, 1, 'same', activation='relu')(Dd1)
Dd1 = layers.Conv2DTranspose(512, 5, 2, 'same')(Dd1)  # Upsampling
Dd1 = layers.BatchNormalization()(Dd1)
Dd1 = layers.Dropout(0.5)(Dd1)
Dd1 = layers.concatenate([Dd1, De7], 3)
# Dd1: (None, 2, 2, 512*2)
Dd2 = layers.Activation('relu')(Dd1)
Dd2 = layers.Conv2D(512, 3, 1, 'same', activation='relu')(Dd2)
Dd2 = layers.Conv2DTranspose(512, 5, 2, 'same')(Dd2)  # Upsampling
Dd2 = layers.BatchNormalization()(Dd2)
Dd2 = layers.Dropout(0.5)(Dd2)
Dd2 = layers.concatenate([Dd2, De6], 3)
# Dd2: (None, 4, 4, 512*2)
Dd3 = layers.Activation('relu')(Dd2)
Dd3 = layers.Conv2D(512, 3, 1, 'same', activation='relu')(Dd3)
Dd3 = layers.Conv2DTranspose(512, 5, 2, 'same')(Dd3)  # Upsampling
Dd3 = layers.BatchNormalization()(Dd3)
Dd3 = layers.Dropout(0.5)(Dd3)
Dd3 = layers.concatenate([Dd3, De5], 3)
# Dd3: (None, 8, 8, 512*2)
Dd4 = layers.Activation('relu')(Dd3)
Dd4 = layers.Conv2D(512, 3, 1, 'same', activation='relu')(Dd4)
Dd4 = layers.Conv2DTranspose(512, 5, 2, 'same')(Dd4)  # Upsampling
Dd4 = layers.BatchNormalization()(Dd4)
Dd4 = layers.Dropout(0.5)(Dd4)
Dd4 = layers.concatenate([Dd4, De4], 3)
# Dd4: (None, 16, 16, 512*2)
Dd5 = layers.Activation('relu')(Dd4)
Dd5 = layers.Conv2D(512, 3, 1, 'same', activation='relu')(Dd5)
Dd5 = layers.Conv2DTranspose(256, 5, 2, 'same')(Dd5)  # Upsampling
Dd5 = layers.BatchNormalization()(Dd5)
Dd5 = layers.Dropout(0.5)(Dd5)
Dd5 = layers.concatenate([Dd5, De3], 3)
# Dd5: (None, 32, 32, 256*2)
Dd6 = layers.Activation('relu')(Dd5)
Dd6 = layers.Conv2D(256, 3, 1, 'same', activation='relu')(Dd6)
Dd6 = layers.Conv2DTranspose(128, 5, 2, 'same')(Dd6)  # Upsampling
Dd6 = layers.BatchNormalization()(Dd6)
Dd6 = layers.Dropout(0.5)(Dd6)
Dd6 = layers.concatenate([Dd6, De2], 3)
# Dd6: (None, 64, 64, 128*2)
Dd7 = layers.Activation('relu')(Dd6)
Dd7 = layers.Conv2D(128, 3, 1, 'same', activation='relu')(Dd7)
Dd7 = layers.Conv2DTranspose(64, 5, 2, 'same')(Dd7)  # Upsampling
Dd7 = layers.BatchNormalization()(Dd7)
Dd7 = layers.Dropout(0.5)(Dd7)
Dd7 = layers.concatenate([Dd7, De1], 3)
# Dd7: (None, 128, 128, 64*2)
Dd8 = layers.Activation('relu')(Dd7)
Dd8 = layers.Conv2D(64, 3, 1, 'same', activation='relu')(Dd8)
Dd8 = layers.Conv2DTranspose(1, 5, 2, 'same')(Dd8)  # Upsampling
Dd8 = layers.Activation('tanh')(Dd8)