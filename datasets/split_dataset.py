import os
import argparse
import shutil


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d',
        '--dataset-path',
        dest='dataset_path',
        help='Which folder to process (it should have subfolders testA, testB, trainA and trainB'
    )
    parser.add_argument(
        '-o',
        '--output_dir',
        dest='output_dir',
        help='Which folder to output'
    )
    args = parser.parse_args()
    return args


def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    dataset_folder = args.dataset_path
    output_root_dir = args.output_dir
    train_a_path = os.path.join(dataset_folder, 'trainA')
    train_b_path = os.path.join(dataset_folder, 'trainB')
    split_dataset(train_a_path, output_root_dir, 'trainA')
    split_dataset(train_b_path, output_root_dir, 'trainB')


def split_dataset(dataset, output_root_dir, dataset_type):
    count = 1
    for root_dir, dir_names, filenames in sorted(os.walk(dataset)):
        filenames.sort()
        for i in range(0, len(filenames), 30):
            if i + 30 > len(filenames):
                break
            output_dir = os.path.join(
                output_root_dir, dataset_type, f'{count:04}'
            )
            while os.path.exists(output_dir):
                count += 1
                output_dir = os.path.join(
                    output_root_dir, dataset_type, f'{count:04}'
                )
            os.makedirs(output_dir)
            print(f"Made dir {output_dir}")
            for filename in filenames[i:i+30]:
                output_name = os.path.join(output_dir, filename)
                filename = os.path.join(root_dir, filename)
                shutil.copyfile(filename, output_name)
                print(f"Copied {filename} to {output_name}")


if __name__ == '__main__':

    args = get_args()
    main(args)
