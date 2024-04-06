import torch
import matplotlib.pyplot as plt

def data_evaluation(class_1_tensors,class_2_tensors):
    # 将每类样本放入一个张量矩阵中
    # class_1_tensor_matrix = torch.stack(class_1_tensors)
    # class_2_tensor_matrix = torch.stack(class_2_tensors)
    class_1_tensor_matrix = class_1_tensors
    class_2_tensor_matrix = class_2_tensors

    # 计算每类样本之间的欧氏距离
    euclidean_distances_class_1 = torch.cdist(class_1_tensor_matrix, class_1_tensor_matrix, p=2)
    euclidean_distances_class_2 = torch.cdist(class_2_tensor_matrix, class_2_tensor_matrix, p=2)
    euclidean_distances_class_1_2 = torch.cdist(class_1_tensor_matrix, class_2_tensor_matrix, p=2)
    
    total_euclidean_distance_1 = torch.sum(euclidean_distances_class_1)
    total_euclidean_distance_2 = torch.sum(euclidean_distances_class_2)
    total_euclidean_distance_1_2 = torch.sum(euclidean_distances_class_1_2)
    # 对总的欧式距离进行归一化，以平衡不同类别样本数量的影响
    num_samples_class_1 = class_1_tensor_matrix.size(0)
    num_samples_class_2 = class_2_tensor_matrix.size(0)
    total_euclidean_distance_normalized_1 = total_euclidean_distance_1 / (num_samples_class_1 * num_samples_class_1)
    total_euclidean_distance_normalized_2 = total_euclidean_distance_2 / (num_samples_class_2 * num_samples_class_2)
    total_euclidean_distance_normalized_1_2 = total_euclidean_distance_1_2 / (num_samples_class_1 * num_samples_class_2)
    # # 使用 Matplotlib 可视化欧式距离矩阵
    # plt.imshow(euclidean_distances_class_1_2, cmap='viridis', interpolation='nearest')
    # plt.colorbar(label='Euclidean Distance')
    # plt.xlabel('Class 2 Samples')
    # plt.ylabel('Class 1 Samples')
    # plt.title('Euclidean Distance Matrix')
    # plt.show()

    # 计算每类样本之间的余弦相似度
    # normalized_class_1_tensor_matrix = class_1_tensor_matrix / torch.norm(class_1_tensor_matrix, dim=1, keepdim=True)
    # cosine_similarities_class_1 = torch.mm(normalized_class_1_tensor_matrix, normalized_class_1_tensor_matrix.t())

    # normalized_class_2_tensor_matrix = class_2_tensor_matrix / torch.norm(class_2_tensor_matrix, dim=1, keepdim=True)
    # cosine_similarities_class_2 = torch.mm(normalized_class_2_tensor_matrix, normalized_class_2_tensor_matrix.t())
    return total_euclidean_distance_normalized_1,total_euclidean_distance_normalized_2,total_euclidean_distance_normalized_1_2