from omegaconf import DictConfig, OmegaConf
import hydra
import pickle
import logging

from src.data.make_dataset import get_mnist_dataloader, get_cifar10_dataloader
from src.models.make_model import get_model
from src.models.train_model import train
from src.models.test_model import test
from src.visualization.visualize import print_results, print_training_evolution

@hydra.main(config_path='./config', config_name='config')
def main(cfg: DictConfig) -> None:
    # setup logging
    log = logging.getLogger('__main__')
    log.setLevel(logging.INFO)
    
    log.info(OmegaConf.to_yaml(cfg))

    log.info('\n------CREATING TRAINING AND TEST DATASET------\n')

    # get loaders for training and test datasets
    if cfg.data.name == 'mnist':
        train_loader = get_mnist_dataloader(cfg, 'train')
        test_loader = get_mnist_dataloader(cfg, 'test')
    else:
        train_loader = get_cifar10_dataloader(cfg, 'train')
        test_loader = get_cifar10_dataloader(cfg, 'test')

    log.info('\n------GENERATING MODEL------\n')

    # generate untrained model
    model = get_model(cfg)

    log.info('\n------STARTING TRAINING------\n')

    # train the model and get loss history
    losses = train(cfg, model, train_loader)

    # save the training evolution for later comparison
    with open('losses.pickle', 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(losses, f, pickle.HIGHEST_PROTOCOL)

    # print training loss evolution
    print_training_evolution(losses)

    # evaluate model on test dataset and get metrics
    cm, accuracy, test_loss = test(model, test_loader)


    # print metrics (confusion matrix, accuracy and loss)
    print_results(cm, accuracy, test_loss)

    return test_loss


if __name__ == "__main__":
    main()