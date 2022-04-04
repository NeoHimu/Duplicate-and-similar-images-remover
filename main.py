from DuplicateRemover import DuplicateRemover

dirname = "images"

# Remove Duplicates
dr = DuplicateRemover(dirname)
# dr.find_duplicates()

# Find Similar Images
# dr.find_similar("images/6.webp",70)
# dr.find_all_clusters(75)
# dr.local_testing_find_all_clusters(75)
# dr.vgg_vectors("images/6.webp")
dr.vgg_all_clusters()