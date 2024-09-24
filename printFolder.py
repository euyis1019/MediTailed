import folder_tree

# basic usage
# output = folder_tree.print_tree(path='..', max_depth=1)
# print(output)


# parameters
output = folder_tree.print_tree(
    path='D:\project\MediTailed\data\PBC_dataset_normal_DIB',
    max_depth=0,
    exclude=['.git', 'coco', 'voc'],
    exclude_patterns=['*.pyc', '__pycache__','*.dll'],
    show_hidden=False,
    include_file_sizes=True,
    output_format='string',
)
print(output)