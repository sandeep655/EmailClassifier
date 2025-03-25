from model.randomforest import RandomForest

def model_predict(data, df, name):
    results = []
    print("RandomForest model for group:", name)
    # model = RandomForest("RandomForest", data.get_embeddings(), data.get_type())
    model = RandomForest("RandomForest", data.get_embeddings(),
                             data.y_chain_train if hasattr(data, 'y_chain_train') else None)
    model.train(data)
    model.predict(data.X_test)
    model.print_results(data)


def model_evaluate(model, data):
    model.print_results(data)