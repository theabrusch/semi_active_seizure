import pickle
import matplotlib.pyplot as plt

with open('/Users/theabrusch/Desktop/Speciale_data/val_loss.pickle', 'rb') as f:
    val_loss = pickle.load(f)

with open('/Users/theabrusch/Desktop/Speciale_data/train_loss.pickle', 'rb') as f:
    train_loss = pickle.load(f)

plt.plot(train_loss)
plt.plot(val_loss)
plt.show()