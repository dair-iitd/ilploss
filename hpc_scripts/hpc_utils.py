import os


def replace_spl(s):
    replace_chars = [" ", "/", "_", "-"]
    for this_char in replace_chars:
        s = s.replace(this_char, ".")
    return s


def parse_filename(s):
    bname = os.path.basename(s)
    bname = ".".join(bname.split(".")[:-1])
    return replace_spl(bname)


def extract_sudoku_size(s):
    file_name = os.path.split(s)[1]
    return file_name.split("_")[1]


def parse_path_to_params(p,**kwargs):
    #p == /home/yatin/softlinks/hpcscratch/comboptnet/comboptnet-data-in-our-format/datasets/static_constraints/set_covering/8_dim/4_const/9/dataset.pt'
    factors = kwargs.get('factors',[1])
    task, dim,const,dseed,_ = p.split(os.sep)[-5:]
    dim = int(dim[:-4])
    const = int(const[:-6])
    dseed = int(dseed)
    rvlist =[]
    for factor in factors:
        logger_dir = os.path.join(*p.split(os.sep)[-6:-1],'{}_lconst'.format(factor*const))
        rvlist.append((p,task,dim,const,factor*const,dseed,logger_dir))
    #return p, task, dim, const, dseed 
    return rvlist

def get_list_of_paramdicts_random_constraints(source_dir,**kwargs):
    is_data_file = lambda x: x.endswith('.pt')
    parameters = []
    for (root,dirs,files)  in os.walk(source_dir):
        if len(files) > 0:
            for this_file in files:
                if is_data_file(this_file):
                    parameters.extend(parse_path_to_params(os.path.join(root, this_file),**kwargs))   
                    
    return parameters

def extract_values_from_path(d):
    values_list = get_list_of_paramdicts_random_constraints(**d['init_args'])
    d['values_list'] = values_list
    return param_groups_default_read_out(d)

def param_groups_default_read_out(d):
    return (d['short_names'],
            d['names'], 
            d['create_dirs'], 
            d['values_list']
            )

def read_file_names_from_dir(dir_name, **kwargs):
    fnames = os.listdir(dir_name)
    base_dir = kwargs.get('base_dir',dir_name)
    fnames = [os.path.join(base_dir, x) for x in fnames]
    return fnames

