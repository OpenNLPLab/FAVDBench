import os
import os.path as op


def ensure_directory(path):
    if path == '' or path == '.':
        return
    if path != None and len(path) > 0:
        assert not op.isfile(path), '{} is a file'.format(path)
        if not os.path.exists(path) and not op.islink(path):
            try:
                os.makedirs(path)
            except:
                if os.path.isdir(path):
                    # another process has done makedir
                    pass
                else:
                    raise


def tsv_writer(values, tsv_file_name, sep='\t'):
    ensure_directory(os.path.dirname(tsv_file_name))
    tsv_lineidx_file = os.path.splitext(tsv_file_name)[0] + '.lineidx'
    tsv_8b_file = tsv_lineidx_file + '.8b'
    idx = 0
    tsv_file_name_tmp = tsv_file_name + '.tmp'
    tsv_lineidx_file_tmp = tsv_lineidx_file + '.tmp'
    tsv_8b_file_tmp = tsv_8b_file + '.tmp'
    import sys
    is_py2 = sys.version_info.major == 2
    if not is_py2:
        sep = sep.encode()
    with open(tsv_file_name_tmp,
              'wb') as fp, open(tsv_lineidx_file_tmp,
                                'w') as fpidx, open(tsv_8b_file_tmp,
                                                    'wb') as fp8b:
        assert values is not None
        for value in values:
            assert value is not None
            if is_py2:
                v = sep.join(
                    map(
                        lambda v: v.encode('utf-8')
                        if isinstance(v, unicode) else str(v), value)) + '\n'
            else:
                value = map(
                    lambda v: v
                    if type(v) == bytes else str(v).encode(), value)
                v = sep.join(value) + b'\n'
            fp.write(v)
            fpidx.write(str(idx) + '\n')
            # although we can use sys.byteorder to retrieve the system-default
            # byte order, let's use little always to make it consistent and
            # simple
            fp8b.write(idx.to_bytes(8, 'little'))
            idx = idx + len(v)
    # the following might crash if there are two processes which are writing at
    # the same time. One process finishes the renaming first and the second one
    # will crash. In this case, we know there must be some errors when you run
    # the code, and it should be a bug to fix rather than to use try-catch to
    # protect it here.
    os.rename(tsv_file_name_tmp, tsv_file_name)
    os.rename(tsv_lineidx_file_tmp, tsv_lineidx_file)
    os.rename(tsv_8b_file_tmp, tsv_8b_file)
