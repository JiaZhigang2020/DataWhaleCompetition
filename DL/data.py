import numpy as np
import torch
from torch_geometric.data import Data, DataLoader
# import ray
from rdkit import Chem
from rdkit.Chem import AllChem
from openbabel import pybel
from abc import ABC, abstractmethod
import re
import requests
import pandas as pd
import os

# 初始化 Ray
# ray.init(ignore_reinit_error=True)


class Graph(ABC):
    def __init__(self):
        self.edge_threshold = 1.6  # 判定存在化学键的距离
        self.atom_type_list = [
            "N",  # 氮原子
            "C",  # 羰基碳原子
            "O",  # 羰基氧原子
        ]
        self.atom_list, self.position_list = [], []

    # @abstractmethod
    def _get_atom_and_position_list(self):
        atom_list, position_list = [], []
        return atom_list, position_list

    def __post_init__(self):
        self.atom_list, self.position_list = self._get_atom_and_position_list()
        self.edge_index_tensor = self.__get_edge_index_tensor()
        self.atom_encode_tensor = self.__get_atom_encode_tensor()

    def __get_edge_index_tensor(self):
        edge_index_list = []
        for first_atom_position_index in range(len(self.position_list)):
            for second_atom_position_index in range(first_atom_position_index + 1, len(self.position_list)):
                distance = torch.norm(torch.tensor(self.position_list[first_atom_position_index]) - torch.tensor(self.position_list[second_atom_position_index]))
                if distance < self.edge_threshold:
                    edge_index_list.append([first_atom_position_index, second_atom_position_index])
                    edge_index_list.append([second_atom_position_index, first_atom_position_index])
        edge_index_tensor = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        return edge_index_tensor

    def __get_atom_encode_tensor(self):
        atom_encode_list = []
        atom_type_list = self.atom_type_list
        for atom_type in self.atom_list:
            atom_encode_template_list = [0] * len(atom_type_list) + [0]
            atom_index = atom_type_list.index(atom_type) if atom_type in atom_type_list else len(atom_type_list)
            atom_encode_template_list[atom_index] = 1
            atom_encode_list.append(atom_encode_template_list)
        atom_encode_tensor = torch.tensor(atom_encode_list, dtype=torch.float).contiguous()
        return atom_encode_tensor

    def get_protein_graph(self):
        protein_graph = Data(x=self.atom_encode_tensor, edge_index=self.edge_index_tensor, dtype=torch.long)
        return protein_graph


# 创建 Ray Actor 并指定资源
# @ray.remote(num_cpus=1, num_gpus=0)
class Protein(Graph):
    def __init__(self, pdb_file_path: str):
        self.pdb_file_path = pdb_file_path
        print(f">>>Processing: {self.pdb_file_path}>>>")
        super().__init__()
        self.__post_init__()

    def _get_atom_and_position_list(self):
        atom_list, position_list = [], []
        with open(self.pdb_file_path, 'r') as f:
            for line in f:
                if line.startswith('ATOM'):
                    item_list = line.strip().split()
                    atom = item_list[2]
                    if atom in self.atom_type_list:
                        try:
                            atom_list.append(atom)
                            value_list = re.findall(r'[-+]?\d*\.\d+|[-+]?\d+', line.strip())
                            position_list.append(np.array(value_list[2:5], dtype=np.float32).tolist())
                        except ValueError as e:
                            print(f'>>>{self.pdb_file_path}-{str(e)}>>>')
        return atom_list, position_list


# @ray.remote(num_cpus=1, num_gpus=0)
class Protac(Graph):
    def __init__(self, smiles):
        print(f">>>Processing: {smiles}>>>")
        self.smiles = smiles
        super().__init__()
        self.__post_init__()

    def _get_atom_and_position_list(self):
        try:
            mol = pybel.readstring("smi", self.smiles)
            mol.addh()  # 添加氢原子
            mol.make3D()  # 生成三维结构
            atom_list = []
            position_list = []
            for atom in mol:
                atom_list.append(atom.type)
                position_list.append(atom.coords)
            return atom_list, position_list
        except Exception as e:
            print(f"Error processing SMILES {self.smiles}: {str(e)}")
            return [], []


class DataFile:
    def __init__(self, file_path: str, batch_size=2, shuffle=True):
        self.file_path = file_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.protein_pdb_folder = ('../dataset/'
                                   'protein_from_alphafold_database/')
        self.data_df = self.__get_file_data_df()
        self.target_name_list, self.e3_ligase_name_list, self.protac_smiles_list, self.label_list = (
            self.__get_complex_compont_name_list())
        self.__get_protein_from_alphafold_database()
        self.protac_smiles_to_graph_dict = self.__get_protac_smiles_to_graph_dict()
        self.target_name_to_graph_dict = self.__get_target_name_to_graph_dict()
        self.e3_ligase_name_to_graph_dict = self.__get_e3_ligase_name_to_graph_dict()

    def __get_file_data_df(self):
        data_df = pd.read_excel(self.file_path)
        return data_df

    def __get_complex_compont_name_list(self):
        target_name_list = self.data_df['Target'].tolist()
        e3_ligase_name_list = self.data_df['E3 ligase'].tolist()
        protac_smiles_list = self.data_df['Smiles'].tolist()
        label_list = self.data_df['Label'].tolist()
        return target_name_list, e3_ligase_name_list, protac_smiles_list, label_list

    def __download_alphafold_structure_from_database(self, uniprot_id, file_name):
        url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
        print(f">>>Downloading {file_name} from {url}")
        try:
            response = requests.get(url)
        except Exception as e:
            print(f">>>{url}-{str(e)}>>>")
        if response.status_code == 200:
            with open(f'{self.protein_pdb_folder}{file_name}', 'wb') as file:
                file.write(response.content)
            return "Successful: {uniprot_id} - {file_name}"
        else:
            return f"Failed: {uniprot_id} - {file_name}"

    def __get_protein_from_alphafold_database(self):
        uniport_and_target_df = self.data_df.loc[:, ["Uniprot", "Target"]].drop_duplicates()
        pdb_file_list = os.listdir(self.protein_pdb_folder)
        for index, row in uniport_and_target_df.iterrows():
            uniprot_id = row["Uniprot"]
            target_id = row["Target"]
            if f"{target_id}.pdb" not in pdb_file_list:
                self.__download_alphafold_structure_from_database(uniprot_id,
                                                                  f"{target_id.replace('/', '-')}.pdb")
        return

    def __get_target_name_to_graph_dict(self):
        unique_target_name_list = pd.unique(self.target_name_list).tolist()
        target_actor_list = [Protein(pdb_file_path=f"../dataset/"
                                                          f"protein_from_alphafold_database/"
                                                          f"{target_name.replace('/', '-')}.pdb")
                             for target_name in unique_target_name_list]

        # 异步获取图数据
        target_graph_futures = [actor.get_protein_graph() for actor in target_actor_list]
        target_graph_list = target_graph_futures

        target_name_to_graph_dict = {unique_target_name_list[index]: target_graph_list[index]
                                     for index in range(len(unique_target_name_list))}
        return target_name_to_graph_dict

    def __get_e3_ligase_name_to_graph_dict(self):
        unique_e3_ligase_name_list = pd.unique(self.e3_ligase_name_list).tolist()
        e3_ligase_actor_list = [Protein(pdb_file_path=f'../dataset/'
                                                             f'protein_from_alphafold_database/{e3_ligase_name}.pdb')
                                for e3_ligase_name in unique_e3_ligase_name_list]

        # 异步获取图数据
        e3_ligase_graph_futures = [actor.get_protein_graph() for actor in e3_ligase_actor_list]
        e3_ligase_graph_list = e3_ligase_graph_futures

        e3_ligase_name_to_graph_dict = {unique_e3_ligase_name_list[index]: e3_ligase_graph_list[index]
                                        for index in range(len(unique_e3_ligase_name_list))}
        return e3_ligase_name_to_graph_dict


    def __get_protac_smiles_to_graph_dict(self):
        unique_protac_smiles_list = pd.unique(self.protac_smiles_list).tolist()
        protac_actor_list = [Protac(protac_smiles) for protac_smiles in unique_protac_smiles_list]

        # 异步获取图数据
        protac_graph_futures = [actor.get_protein_graph() for actor in protac_actor_list]
        protac_graph_list = protac_graph_futures

        protac_smiles_to_graph_dict = {unique_protac_smiles_list[index]: protac_graph_list[index]
                                       for index in range(len(unique_protac_smiles_list))}
        return protac_smiles_to_graph_dict


    def get_data_list(self):
        data_list = []
        target_name_list, e3_ligase_name_list, protac_smiles_list, label_list = (
            self.target_name_list, self.e3_ligase_name_list, self.protac_smiles_list, self.label_list)
        for index in range(len(target_name_list)):
            target_graph = self.target_name_to_graph_dict[target_name_list[index]]
            e3_ligase_graph = self.e3_ligase_name_to_graph_dict[e3_ligase_name_list[index]]
            protac_graph = self.protac_smiles_to_graph_dict[protac_smiles_list[index]]
            data = Data(target=Data(x=target_graph.x, edge_index=target_graph.edge_index),
                        e3_ligase=Data(x=e3_ligase_graph.x, edge_index=e3_ligase_graph.edge_index),
                        protac=Data(x=protac_graph.x, edge_index=protac_graph.edge_index),
                        y=torch.tensor([label_list[index]], dtype=torch.float)
                        )
            data_list.append(data)
        return data_list

    def get_data_loader(self):
        data_list = self.get_data_list()
        print(data_list)
        data_loader = DataLoader(data_list, batch_size=self.batch_size, shuffle=self.shuffle)
        return data_loader


if __name__ == '__main__':
    # dataFile = DataFile(file_path='/mnt/nas/Project_document/DataWhaleCompetition/dataset/traindata-new.xlsx',
    #                     batch_size=1, shuffle=True)
    dataFile = DataFile(file_path='../dataset/traindata-new.xlsx',
                        batch_size=1, shuffle=True)
    dataList = dataFile.get_data_list()
    print(dataList)
    dataLoader = dataFile.get_data_loader()
    print(dataLoader)

