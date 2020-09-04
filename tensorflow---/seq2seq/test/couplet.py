
from model import Model

m = Model(
'/Users/zuobangbang/Desktop/data/train/in.txt',
'/Users/zuobangbang/Desktop/data/train/out.txt',
'/Users/zuobangbang/Desktop/data/test/in.txt',
'/Users/zuobangbang/Desktop/data/test/out.txt',
'/Users/zuobangbang/Desktop/data/vocab',

        num_units=1024, layers=4, dropout=0.2,
        batch_size=32, learning_rate=0.001,
        output_dir='',
        restore_model=False)

m.train(5000000)