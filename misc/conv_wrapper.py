import conv as  conv
import os
import conv
import shlex
source_dir = '/home/cse/phd/csz178057/hpcscratch/comboptnet/comboptnet-data/datasets/static_constraints/dense_random'
destination_dir = '/home/cse/phd/csz178057/hpcscratch/comboptnet/comboptnet-data-in-our-format/datasets/static_constraints/dense_random'
should_convert = lambda x: x.endswith('.p')

exceptions =[]
commands = []
for (root,dirs,files)  in os.walk(source_dir):
    if len(files) > 0:
        for this_file in files:
            if should_convert(this_file):
                ifile = os.path.join(root, this_file)
                ofile = ifile.replace(source_dir, destination_dir) + 't'
                dest_root_dir = os.path.dirname(ofile)
                if not os.path.exists(dest_root_dir):
                    os.makedirs(dest_root_dir)
                #
                dense_flag = '--dense' if 'dense' in ifile else ''
                cmd_str = 'comboptnet {} {} {}'.format(dense_flag, ifile,ofile)
                commands.append(cmd_str)
                conv.main(shlex.split(cmd_str))




