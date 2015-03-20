# -*- coding: utf-8 -*-

import re
import os
import os.path
import stat
import shutil
from traceback              import print_exc


def listdirtype(path, ext):

    lfiles = os.listdir(path)

    i = 0
    while i < len(lfiles):

        fname,fext = os.path.splitext(lfiles[i])

        if fext != '.csv':
            #print 'deleting {0} with fext = {1}'.format(lfiles[i], fext)
            del lfiles[i]
        else :
            i += 1

    return lfiles

def walk_files(top, ext = None, join = False, explore_subfolders = False):
    """
    Walks through a folder and extract all files of a given extension within that folder.
    If extension is None, then all files are returned

    Parameters
    ----------
    top:        string
                The folder from wich to start the search

    ext:        string or None
                Filters out all filed that are not of extension "ext".  If None, then no filter is applied

    join:       Boolean [default = False]
                If true, then the path to each file is joined to each file names.  If false, only the file names  are returned

    explore_subfolders: Boolean [default = False]
                If true, all subfolders of top are explored.  Otherwise, only the files in the top directory are explored.

    Returns
    -------
    retval:     list of strings
                A list of all files located in the top folder or one of its sub folder.  Includes path information if join == True.
    """

    retval = []

    nb_char_ext = len(ext) if ext else None

    for (path, folders, files) in os.walk(top):

        if ext != None:
            files = [fn for fn in files if ext == fn[-nb_char_ext:]]

        if join:
            retval += [os.path.join(path,fn) for fn in files]
        else:
            retval += files

        if explore_subfolders == False:
            break

    return retval


def copytree(org, dst, own=None):
    if os.path.isdir(dst):
        os.rmdir(dst)
    shutil.copytree(org, dst)
    if own != None:
        try :
            from pwd import getpwnam
            os.chmod(dst, stat.S_IRWXU | stat.S_IRWXG | stat.S_IROTH | stat.S_IXOTH)
            os.chown(dst, getpwnam(own).pw_uid, getpwnam(own).pw_gid)
            for root, dirs, files in os.walk(dst):
                for d in dirs:
                    os.chmod(os.path.join(root, d), stat.S_IRWXU | stat.S_IRWXG | stat.S_IROTH | stat.S_IXOTH)
                    os.chown(os.path.join(root, d), getpwnam(own).pw_uid, getpwnam(own).pw_gid)
                for fn in files:
                    os.chmod(os.path.join(root, fn), stat.S_IRWXU | stat.S_IRWXG | stat.S_IROTH | stat.S_IXOTH)
                    os.chown(os.path.join(root, fn), getpwnam(own).pw_uid, getpwnam(own).pw_gid)
        except:
            pass



def flatten(top, ext = None):
    """
    Walks through a directory structure and extract all files of a given extension, then flattens all file names a such that if a
    file is located at :
        top_head/top_tail/somefolder/somefolder/image_x.tif,
    where (top_head, top_tail = os.path.split(top) new name will be:
        top_tail_somefolder_somefolder_image_x.tif

    Returns
    -------
    lOut:   list of strings
            A list of all files located in the top folder or one of its sub folder.
    lfn:    list of strings
            A list of all corresponding file paths as returned by walk_files

    """

    lfn = walk_files(top, ext)
    top_head = os.path.split(top)[1]
    lflat = []
    for fn in lfn:
        newfn = ''
        head = fn
        while True:
            head, tail = os.path.split(head)
            newfn = (tail if newfn == '' else tail + '_') + newfn
            if head == '' or head == '..' or tail == top_head: break
        lflat.append(newfn)
    return lflat, lfn


def walk_folders(top, join = True, filter = None, explore_subfolders = True):
    """
    Returns a sorted list of all folders within a top folder.  The list excludes the 'top' folder.

    Parameters
    ----------
    top:        string
                The top directory from wich to start the search

    filter:     string or None [default=None]
                If none, all folders found are returned.  Othewise, if filter is in the folder name, then the folder is kept otherwise
                it is filtered out.

    join:       Boolean [default = True]
                If true, then base path to each folder is included in the returned list of paths.  Otherwise only the list of folder names
                is returned.

    explore_subfolders: Boolean [default = true]
                If true, all subfolders of top are also explored.  Otherwise, only the files in the top directory are explored.

    Returns
    -------
    retval: list of strings
            The list of folders found
    """

    retval = []

    for (path, folders, files) in os.walk(top):

        if filter != None:
            folders = [folder for folder in folders if filter in folder]

        if join:
            retval += [os.path.join(path, folder) for folder in folders]
        else:
            retval += folders

        if explore_subfolders == False:
            break

    return retval
    
    
def findleafs(root,  filter = None, join=False, sort=False):
    """
    Explore the directory tree from root and find directories leaf directories, that is directories that do not 
    contain directories.
    
    Parameters
    ----------
    
    root:   string
            Root path from which to start search.  If root is itself a leaf, then it is returned
            
    filter: string or None [default=None]
            If none, all leafs are returned.  Othewise, only folders with "filter" in it's name are returned

    join:   bool
            Join root to returned leaf paths

    sort:   bool
            sort returned leaf path
    
    Returns
    -------
    
    leafs:  list
            List of leaf directories
    """
    
    leafs = []
    
    for (path, folders, files) in os.walk(root):

        if filter != None:
            if filter not in os.path.split(path)[1]:
                continue
        
        if not join:
            path = path.replace(root+os.pathsep, '')   
            
        leafs += [path]
        
            
    if sort :
        leafs.sort()


    return leafs
        
        
                    


def make_dir(paths, own=None):
    """
    Create directory from lPath.  If lpath it is a list, then all paths represented in the list are created.  If it is a
    string, then the path represented by this string is used.

    Parameters
    ----------
    paths : list or string

    ownid : string [optional]
            owner of directory.  Default : None

    Returns
    -----
    created:    list or string
                new directory successfully created
    """

    __output_str = True if isinstance(paths, str) else False
    created = []

    if isinstance(paths, str):
        paths = [paths]
        for path in paths:
            if path == '':
                continue
            try:
                os.makedirs(os.path.abspath(path))
                created += [path]
            except OSError as exception:
                if exception.errno != 17 :
                    raise

    if __output_str:
        created = created[0] if created else ''

    return created


def delete (paths):
    """
    Delete a list of file or directories

    Parameters
    ----------
    paths:  Can be a list of string or a string.
            If it is a list, then the files of all paths represented in the list are deleted
    """

    if type(paths) is str:
        paths = [paths]

    for path in paths:
        try:
            shutil.rmtree(path)
        except:
            pass




def delete_folder_content(paths):
    """
    Delete all files contained in each folder contained in the list "path".  The folder is not deleted (only its content)

    Parameters
    ----------
    lPath:  Can be a list of string or a string.
            If it is a list, then the files of all paths represented in the list are deleted
    """

    delete(paths)
    make_dir(paths)


def split(lpath):
    """
    Applies the os.path.split() function on a list of paths and returns two separate arrays

    Parameters
    ----------
    lpath:  list of paths

    Returns
    ----------
    lhead :    list of strings
               array of heads of corresponding to lpath

    ltail :    list os strings
               array of tails of corresponding to lpath
    """

    lhead = []
    ltail = []
    for path in lpath:
        head, tail = os.path.split(path)
        lhead.append(head)
        ltail.append(tail)

    return (lhead, ltail)

