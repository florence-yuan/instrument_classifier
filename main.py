import modeling

model = modeling.Model()
model.load_model()
model.predict("assets/musicnet/audio/2075.wav")
