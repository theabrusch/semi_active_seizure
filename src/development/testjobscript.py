import argparse

def main(args):
    print('Run ID', args.jobid)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobid', type = str)
    args = parser.parse_args()
    main(args)
