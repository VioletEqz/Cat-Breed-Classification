from Model import Network
from DataIO import preprocess, load_data

if __name__ == '__main__':

    num_epochs = 5
    learning_rate = 0.005
    validate = 1
    verbose = 1
    plot_weights = 1
    plot_correct = 0
    plot_missclassified = 0
    plot_feature_maps = 0

    print('\nLoading dataset...')                 # load dataset
    dataset = load_data()
    dataset = preprocess(dataset)

    print('\nForming Model...')                                   # build model
    model = Network()
    model.build_model()


    print('\nTraining:')                                   # train model
    model.train(
        dataset,
        num_epochs,
        learning_rate,
        validate,
        0,
        plot_weights,
        verbose,
        print_cycle=250
    )

    print('\nTesting:')                                    # test model
    model.evaluate(
        dataset['test_images'],
        dataset['test_labels'],
        0,
        plot_correct,
        plot_missclassified,
        plot_feature_maps,
        verbose
    )
    
    print('\nSaving Model...')
    model.save('model.pt')
    print('\nExiting.')