import pickle


def read_pickles(file: str):
    with open(file, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    print(read_pickles("models.pkl").to_string())