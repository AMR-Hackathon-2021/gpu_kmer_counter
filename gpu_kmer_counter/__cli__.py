import argparse
import psutil 

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input',dest='input',help='Provide the path to the input directory with FAST[AQ] files.')
    parser.add_argument('-o','--output',dest='out', default="out",help='The name used for the output results.')
    parser.add_argument('-k','--k-size',dest='ksize',default=25,help='The k-mer length. Should be a positive integer.')
    parser.add_argument('-K','--k-guess',dest='kguess',default=5*10^6,help='The number of approximate k-mers in the input files.')
    parser.add_argument('-m','--avail-memory',dest='mem',default=psutil.virtual_memory().available/(1024.0 ** 3),help='The available memory to use the tool. Should be in GB')
    parser.add_argument('-f','--false-positive-rate',dest='fpr',default=0.0,help='Set the allowable false positive rate.')

    args = parser.parse_args()

    def extract_cli():
        seq_dir = args.input
        k = args.ksize
        k_guess = args.kguess
        mem = args.mem
        fpr = args.fpr

        if not isinstance(k, int) or not isinstance(fpr,float):
            sys.exit(0)
        else:
            print(
"""
Testing to see if values can be extracted: 
{}
{}
{}
{}
{}
""".format(seq_dir,k,k_guess,mem,fpr))



if __name__ == '__cli__':
    cli()