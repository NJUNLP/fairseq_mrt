# 过滤空行
import argparse

def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="skip blank"
    )
    parser.add_argument(
        '--lang', '-l', type=str, 
        help="lang")
    parser.add_argument(
        '--input_prefix', '-i', type=str, 
        help="input")
    parser.add_argument(
        '--output_prefix', '-o', type=str, 
        help="output")
    return parser

def skip_blank(lang, input_prefix, output_prefix):
    l1, l2 = lang.split('-')
    num = 0
    s = 0
    with open(input_prefix + '.' + l1, 'r', encoding='utf-8') as f1, open(input_prefix + '.' + l2, 'r', encoding='utf-8') as f2, \
        open(output_prefix + '.' + l1, 'w', encoding='utf-8') as s1, open(output_prefix + '.' + l2, 'w', encoding='utf-8') as s2:
        for line1, line2 in zip(f1.readlines(), f2.readlines()):
            s += 1
            if line1.strip('\n') != '' and line2.strip('\n') != '':
                s1.write(line1)
                s2.write(line2)
                num += 1
    print('all lines: %d' % s)
    print('save lines: %d' %num)

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    skip_blank(args.lang, args.input_prefix, args.output_prefix)