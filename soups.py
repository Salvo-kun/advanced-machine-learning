
import sys
import torch
import logging
import os
import test
import options
import commons
from model import network
from datasets.test_dataset import TestDataset
from datetime import datetime

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True  # Provides a speedup

    args = options.parse_arguments(is_training=False)
    start_time = datetime.now()
    output_folder = f"logs/{args.save_dir}/{start_time.strftime('%Y-%m-%d_%H-%M-%S')}"
    commons.make_deterministic(args.seed)
    commons.setup_logging(output_folder, console="info")
    logging.info(" ".join(sys.argv))
    logging.info(f"Arguments: {args}")
    logging.info(f"The outputs are being saved in {output_folder}")

    NUM_MODELS = len(options.model_paths)

    if options.aggregate_by == 'uniform_soup':
        logging.info(f'Performing an uniform soup over {NUM_MODELS} models in this order (paths follows): {options.model_paths}')

        for j, model_path in enumerate(options.model_paths):
            logging.info(f'Adding model located at {model_path} ({j+1} out of {NUM_MODELS}) to uniform soup.')
            assert os.path.exists(model_path)
            state_dict = torch.load(model_path)

            if j == 0:
                uniform_soup = {k : v * (1./NUM_MODELS) for k, v in state_dict.items()}
            else:
                uniform_soup = {k : v * (1./NUM_MODELS) + uniform_soup[k] for k, v in state_dict.items()}

        torch.save(uniform_soup, f'{output_folder}/uniform_soup_N{NUM_MODELS}.pth')
    elif options.aggregate_by == 'greedy_soup':
        logging.info(f'Performing a greedy soup over {NUM_MODELS} models on dataset {args.dataset_folder} in this order (paths follows): {options.model_paths}')
        best_recalls = None
        greedy_soup_ingredients = []
        greedy_soup_params = dict()

        for i, path in enumerate(options.model_paths):
            if i == 0:
                logging.info(f'Init with model located at {path} ({i+1} out of {NUM_MODELS}) to greedy soup.')
                new_ingredient_params = [path]
                potential_greedy_soup_params = torch.load(path)
            else:
                logging.info(f'Trying adding model located at {path} ({i+1} out of {NUM_MODELS}) to greedy soup.')
                new_ingredient_params = torch.load(path)
                num_ingredients = len(greedy_soup_ingredients)
                potential_greedy_soup_params = {
                    k : greedy_soup_params[k].clone() * (num_ingredients / (num_ingredients + 1.)) + 
                        new_ingredient_params[k].clone() * (1. / (num_ingredients + 1))
                    for k in new_ingredient_params
                }

            model = network.GeoLocalizationNet(args.backbone, args.fc_output_dim)
            model.load_state_dict(potential_greedy_soup_params)
            model.to(args.device)
            test_dataset = TestDataset(args.test_set_folder, queries_folder='queries_v1', positive_dist_threshold=args.positive_dist_threshold)
            new_recalls, _ = test.test(args, test_dataset, model)

            # If accuracy on the held-out val set increases, add the new model to the greedy soup.
            logging.info(f'Potential greedy soup recalls {new_recalls}, best so far {best_recalls}.')
            if best_recalls is None or new_recalls > best_recalls:
                greedy_soup_ingredients.append(path)
                best_recalls = new_recalls
                greedy_soup_params = potential_greedy_soup_params
                logging.info(f'Adding to greedy soup. New greedy soup is {greedy_soup_ingredients} with recalls {new_recalls}')

        logging.info(f'Best greedy soup on {args.dataset_folder}: {greedy_soup_ingredients} with recalls {best_recalls}')
        torch.save(greedy_soup_params, f'{output_folder}/greedy_soup_N{NUM_MODELS}.pth')
    else:
        is_greedy = options.aggregate_by != 'uniform_ensemble'
        logging.info(f'Performing a {"greedy" if is_greedy else "uniform"} ensemble over {NUM_MODELS} models on dataset {args.dataset_folder} in this order (paths follows): {options.model_paths}')
        descriptors = None
        models = list()
        args.save_descriptors = True 
        best_recalls = None

        for i, path in enumerate(options.model_paths):
            if i == 0:
                logging.info(f'Calculate descriptors of model located at {path} ({i+1} out of {NUM_MODELS}) to init the {"greedy" if is_greedy else "uniform"} ensemble.')
                model = network.GeoLocalizationNet(args.backbone, args.fc_output_dim)
                model.load_state_dict(path)
                model.to(args.device)
                test_dataset = TestDataset(args.test_set_folder, queries_folder='queries_v1')
                test.test(args, test_dataset, model, output_folder)
                descriptors = torch.load(f'{output_folder}/descriptors.pth')
            else:
                logging.info(f'Calculate descriptors of model located at {path} ({i+1} out of {NUM_MODELS}) to be added to {"greedy" if is_greedy else "uniform"} ensemble.')
                model = network.GeoLocalizationNet(args.backbone, args.fc_output_dim)
                model.load_state_dict(path)
                model.to(args.device)
                test_dataset = TestDataset(args.test_set_folder, queries_folder='queries_v1')
                test.test(args, test_dataset, model, output_folder) 
                descriptors += torch.load(f'{output_folder}/descriptors.pth')

            test_dataset = TestDataset(args.test_set_folder, queries_folder='queries_v1')
            predictions = test.get_predictions(test_dataset, args.fc_output_dim, descriptors/(float(i+1)))   
            new_recalls = tuple(test.evaluate_predictions(predictions, test_dataset) / test_dataset.queries_num * 100)

            if best_recalls is None or (new_recalls > best_recalls or not is_greedy):
                models.append(path)
                best_recalls = new_recalls
                best_descriptors = descriptors
                logging.info(f'Adding to {"greedy" if is_greedy else "uniform"} ensemble. New {"greedy" if is_greedy else "uniform"} ensemble is {models} with recalls {new_recalls}')

        logging.info(f'Best {"greedy" if is_greedy else "uniform"} ensemble on {args.dataset_folder}: {models} with recalls {best_recalls}')
        best_descriptors = best_descriptors / float(NUM_MODELS)
        torch.save(best_descriptors, f'{output_folder}/descriptors_{NUM_MODELS}.pth')