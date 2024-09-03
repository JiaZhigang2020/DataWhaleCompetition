import warnings
import numpy as np
from pandarallel import pandarallel
from rdkit.Chem import MACCSkeys, AllChem, rdMolDescriptors
import requests
from rdkit import Chem, rdBase
import pandas as pd
import sqlite3
import psycopg2
from sklearn.feature_extraction.text import TfidfVectorizer
import re

rdBase.DisableLog('rdApp.warning')


class SqliteDatabase:
    def __init__(self, database_file_path='./dataset/Sqlite3.db'):
        self.database_file_path = database_file_path

    def create_tables(self, sql_template):
        with sqlite3.connect(self.database_file_path) as connection:
            cursor = connection.cursor()
            cursor.execute(sql_template)
            connection.commit()

    def insert_data(self, sql_template):
        try:
            with sqlite3.connect(self.database_file_path) as connection:
                cursor = connection.cursor()
                cursor.execute(sql_template)
                connection.commit()
        except sqlite3.OperationalError:
            warnings.warn(f"Warning: {sql_template}")

    def select_data(self, sql_template):
        with sqlite3.connect(self.database_file_path) as connection:
            cursor = connection.cursor()
            cursor.execute(sql_template)
            data = cursor.fetchall()[0][0]
        return data

    def delete_data(self, sql_template):
        with sqlite3.connect(self.database_file_path) as connection:
            cursor = connection.cursor()
            cursor.execute(sql_template)
            connection.commit()


class PostgresDatabase:
    def __init__(self, database_url="postgresql://postgres:postgres@192.168.31.200:5432/postgres"):
        # PostgreSQL 使用的是连接字符串，而不是文件路径
        self.database_url = database_url

    def _get_connection(self):
        # 创建数据库连接
        return psycopg2.connect(self.database_url)

    def create_tables(self, sql_template):
        try:
            with self._get_connection() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(sql_template)
                    connection.commit()
        except psycopg2.Error as e:
            print(f"Error creating tables: {e}")
            connection.rollback()

    def insert_data(self, sql_template):
        try:
            with self._get_connection() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(sql_template)
                    connection.commit()
        except psycopg2.Error as e:
            warnings.warn(f"Warning: {sql_template}")
            print(f"Error inserting data: {e}")
            connection.rollback()

    def select_data(self, sql_template):
        try:
            with self._get_connection() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(sql_template)
                    data = cursor.fetchall()
                    return data[0][0]
        except psycopg2.Error as e:
            print(f"Error selecting data: {e}")
            return None

    def delete_data(self, sql_template):
        try:
            with self._get_connection() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(sql_template)
                    connection.commit()
        except psycopg2.Error as e:
            print(f"Error deleting data: {e}")
            connection.rollback()


class DataFile:
    def __init__(self, training_data_file_path, testing_data_file_path, database=PostgresDatabase(), num_workers=8):
        self.training_data_file_path = training_data_file_path
        self.testing_data_file_path = testing_data_file_path
        self.database = database
        self.num_workers = num_workers
        self.training_data_df = self.__get_data_df(self.training_data_file_path)  # 训练用的文件数据
        self.training_data_df = self.__data_pretreatment(self.training_data_df)  # 训练集数据添加标签
        self.testing_data_df = self.__get_data_df(self.testing_data_file_path)  # 预测用的文件数据
        self.training_data_df = self.__delete_test_data(self.training_data_df, self.testing_data_df)  # 删除训练集中混合的测试集数据
        self.uniprot_id_to_num_dict = self.__get_uniprot_id_to_num_dict()
        self.target_to_num_dict = self.__get_target_to_num_dict()
        self.e3_ligase_to_num_dict = self.__get_e3_ligase_to_num_dict()
        self.targeting_ligand_and_e3_ligase_recruiter_to_num_dict = (
            self.__get_targeting_ligand_and_e3_ligase_recruiter_to_num_dict())
        self.vectorizer = TfidfVectorizer()  # TF-IDF编码字典

    def __data_pretreatment(self, data_df: pd.DataFrame) -> pd.DataFrame:
        if "Label" in data_df.columns:
            print(data_df['Label'].value_counts())
            return data_df
        else:
            label_list = []

            def get_dc50_label(dc50_string):
                if not pd.isna(dc50_string):
                    num_list = re.findall(r"\d+\.\d*|\d+", dc50_string)
                    if len(num_list) > 0:
                        for num in num_list:
                            if float(num) <= 100:
                                return 1
                return 0

            def get_dmax_label(dmax_string):
                if not pd.isna(dmax_string):
                    num_list = re.findall(r"\d+\.\d*|\d+", dmax_string)
                    if len(num_list) > 0:
                        for num in num_list:
                            if float(num) >= 80:
                                return 1
                return 0

            dc50_label_list = data_df.loc[:, "DC50 (nM)"].apply(lambda x: get_dc50_label(str(x)))
            dmax_label_list = data_df.loc[:, "Dmax (%)"].apply(lambda x: get_dmax_label(str(x)))
            for index in range(len(dc50_label_list)):
                if dc50_label_list[index] == 1 or dmax_label_list[index] == 1:
                    label_list.append(1)
                else:
                    label_list.append(0)
        data_df['Label'] = label_list
        print(data_df['Label'].value_counts())
        return data_df

    def __delete_test_data(self, traing_data_df, testing_data_df):
        testing_smiles_series = testing_data_df['Smiles']
        traing_data_df = traing_data_df[~traing_data_df['Smiles'].isin(testing_smiles_series)]
        traing_data_df.reset_index(drop=True, inplace=True)
        return traing_data_df


    def __get_data_df(self, data_file_path) -> pd.DataFrame:
        data_df = pd.read_excel(data_file_path)
        return data_df

    def __get_topological_fingerprints_encode(self, smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        encoded_smiles = list(Chem.RDKFingerprint(mol))
        return encoded_smiles

    def __get_maccs_encode(self, smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        encoded_smiles = list(MACCSkeys.GenMACCSKeys(mol))
        return encoded_smiles

    def __get_atom_pairs_encode(self, smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        # encoded_smiles = list(Pairs.GetAtomPairFingerprint(mol))
        encoded_smiles = list(rdMolDescriptors.GetHashedAtomPairFingerprint(mol, nBits=256))
        return encoded_smiles

    def __get_targeting_ligand_and_e3_ligase_recruiter_to_num_dict(self):
        targeting_ligand_and_e3_ligase_recruiter_to_num_dict = {}
        for temp in self.training_data_df['Smiles'].apply(lambda x: x[:20]):
            if temp not in targeting_ligand_and_e3_ligase_recruiter_to_num_dict.keys():
                targeting_ligand_and_e3_ligase_recruiter_to_num_dict[temp] = (
                        len(targeting_ligand_and_e3_ligase_recruiter_to_num_dict.keys()) + 1)
        for temp in self.training_data_df['Smiles'].apply(lambda x: x[-20:]):
            if temp not in targeting_ligand_and_e3_ligase_recruiter_to_num_dict.keys():
                targeting_ligand_and_e3_ligase_recruiter_to_num_dict[temp] = (
                        len(targeting_ligand_and_e3_ligase_recruiter_to_num_dict.keys()) + 1)
        return targeting_ligand_and_e3_ligase_recruiter_to_num_dict

    def __get_targeting_ligand_and_e3_ligase_recruiter_encode(self, smiles_string):
        try:
            left_smiles_encode = self.targeting_ligand_and_e3_ligase_recruiter_to_num_dict[smiles_string[:20]]
        except KeyError as e:
            left_smiles_encode = 0
        try:
            right_smiles_encode = self.targeting_ligand_and_e3_ligase_recruiter_to_num_dict[smiles_string[:-20]]
        except KeyError as e:
            right_smiles_encode = 0
        return [left_smiles_encode, right_smiles_encode]

    def get_targeting_ligand_and_e3_ligase_recruiter_encoded_series(self, data_df):
        targeting_ligand_and_e3_ligase_recruiter_encoded = (
            data_df['Smiles'].apply(self.__get_targeting_ligand_and_e3_ligase_recruiter_encode))
        return targeting_ligand_and_e3_ligase_recruiter_encoded

    def __get_morgan_fingerprints_encode(self, smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        encoded_smiles = list(AllChem.GetMorganFingerprintAsBitVect(mol, 2))
        return encoded_smiles

    def get_smiles_encoded_series(self, data_df, encode_function):
        # print("Getting SMILES encoded")
        pandarallel.initialize(nb_workers=self.num_workers, progress_bar=False, verbose=0)
        smiles_encoded_series = data_df['Smiles'].parallel_apply(encode_function)
        return smiles_encoded_series

    def __get_protein_from_uniprot(self, protein_id):
        print(f'Get Protein Sequence: {protein_id}')
        protein_sequence = ''
        # 从uniprot 中获取蛋白质序列
        headers = {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Encoding': 'gzip, deflate, br, zstd',
            'Accept-Language': 'en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7',
            'Cache-Control': 'max-age=0',
            'Cookie': '_hjSessionUser_2638207=eyJpZCI6ImI1NDVlMmQ1LTA2OGMtNWE1Ni05N2M5LTAwYmJjODM0MDlhZCIsImNyZWF0ZWQiOjE3MjA4Mzg5NDMzNDQsImV4aXN0aW5nIjp0cnVlfQ==; _ga=GA1.2.1417426880.1720838941; _ga_V6TXEC4BDF=GS1.1.1720945124.2.0.1720945124.0.0.0',
            'Priority': 'u=0, i',
            'Sec-Ch-Ua': '"Not/A)Brand";v="8", "Chromium";v="126", "Microsoft Edge";v="126"',
            'Sec-Ch-Ua-Mobile': '?0',
            'Sec-Ch-Ua-Platform': '"macOS"',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Upgrade-Insecure-Requests': '1',
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 Edg/126.0.0.0'
        }
        url = f'https://rest.uniprot.org/uniprotkb/{protein_id}.fasta'
        response = requests.get(url, headers=headers)
        uniprot_response = response.content.decode('utf-8')
        uniprot_line_list = uniprot_response.split('\n')
        for sequence in uniprot_line_list[1:]:
            protein_sequence += sequence
        return protein_sequence

    def __get_protein_sequence_list(self, data_df: pd.DataFrame):
        protein_sequence_list = []
        self.database.create_tables(sql_template="""
            CREATE TABLE IF NOT EXISTS "public"."ProteinSequence" (
                "ProteinID" varchar(100) PRIMARY KEY,
                "Sequence" text NOT NULL
            )
        """)
        protein_id_series = data_df['Uniprot']
        for index, protein_id in enumerate(protein_id_series):
            if pd.isnull(protein_id):
                protein_sequence_list.append(None)
            else:
                try:
                    protein_sequence = self.database.select_data(
                        f'SELECT "Sequence" from "public"."ProteinSequence" WHERE '
                        f'"ProteinID" = \'{protein_id}\'')
                except IndexError:
                    protein_sequence = self.__get_protein_from_uniprot(protein_id)
                    self.database.insert_data(
                        sql_template=f'insert into "public"."ProteinSequence" ("ProteinID", "Sequence") '
                                     f'values (\'{protein_id}\', \'{protein_sequence}\')')
                protein_sequence_list.append(protein_sequence)
        return protein_sequence_list

    def __get_single_protein_sequence_AAC_encode(self, single_protein_sequence):
        aac_encode_list = [0] * 20  # order: ACDEFGHIKLMNPQRSTVWY
        aac_list = ' '.join('ACDEFGHIKLMNPQRSTVWY').split()
        try:
            protein_squence_length = len(single_protein_sequence)
        except TypeError:
            return aac_encode_list
        for amino_acid in single_protein_sequence:
            try:
                index = aac_list.index(amino_acid)
                aac_encode_list[index] += 1
            except ValueError:
                protein_squence_length -= 1
        aac_encode_list = (np.array(aac_encode_list) / protein_squence_length).tolist()
        return aac_encode_list

    def get_protein_sequence_encode_Series(self, data_df: pd.DataFrame, single_protein_sequence_encode_function):
        protein_sequence_list = self.__get_protein_sequence_list(data_df)
        protein_sequence_series = pd.Series(protein_sequence_list)
        pandarallel.initialize(nb_workers=self.num_workers, progress_bar=False, verbose=0)
        protein_sequence_encode_series = protein_sequence_series.parallel_apply(single_protein_sequence_encode_function)
        return protein_sequence_encode_series

    def __get_uniprot_id_to_num_dict(self):
        uniprot_id_to_num_dict = {}
        for uniprot in self.training_data_df['Uniprot']:
            if uniprot not in uniprot_id_to_num_dict.keys():
                uniprot_id_to_num_dict[uniprot] = len(uniprot_id_to_num_dict.keys()) + 1
        return uniprot_id_to_num_dict

    def __get_uniprot_id_encode(self, uniprot_id):
        try:
            encoded_uniprot_id = self.uniprot_id_to_num_dict[uniprot_id]
        except KeyError as e:
            encoded_uniprot_id = 0
        return [encoded_uniprot_id]

    def get_uniprot_id_encoded_series(self, data_df):
        uniprot_encoded_series = data_df['Uniprot'].apply(self.__get_uniprot_id_encode)
        return uniprot_encoded_series

    def __get_target_to_num_dict(self):
        target_to_num_dict = {}
        for uniprot in self.training_data_df['Target']:
            if uniprot not in target_to_num_dict.keys():
                target_to_num_dict[uniprot] = len(target_to_num_dict.keys()) + 1
        return target_to_num_dict

    def __get_target_encode(self, target):
        try:
            target_encode = self.target_to_num_dict[target]
        except KeyError as e:
            target_encode = 0
        return [target_encode]

    def get_target_encoded_series(self, data_df):
        target_encoded_series = data_df['Target'].apply(self.__get_target_encode)
        return target_encoded_series

    def __get_e3_ligase_to_num_dict(self):
        e3_ligase_to_num_dict = {}
        for uniprot in self.training_data_df['E3_ligase']:
            if uniprot not in e3_ligase_to_num_dict.keys():
                e3_ligase_to_num_dict[uniprot] = len(e3_ligase_to_num_dict.keys()) + 1
        return e3_ligase_to_num_dict

    def __get_e3_ligase_encode(self, e3_ligase):
        try:
            e3_ligase_encode = self.e3_ligase_to_num_dict[e3_ligase]
        except KeyError as e:
            e3_ligase_encode = 0
        return [e3_ligase_encode]

    def get_e3_ligase_encoded_series(self, data_df):
        e3_ligase_encoded_series = data_df['E3_ligase'].apply(self.__get_e3_ligase_encode)
        return e3_ligase_encoded_series

    def get_assay_tf_idf_encoded_series(self, data_df, is_train=True):
        data_df['Assay (DC50/Dmax)'][pd.isna(data_df['Assay (DC50/Dmax)'])] = 'NoData'
        if is_train:
            assay_tf_idf_encoded_array = self.vectorizer.fit_transform(data_df['Assay (DC50/Dmax)'].astype('str')).toarray()
        else:
            assay_tf_idf_encoded_array = self.vectorizer.transform(data_df['Assay (DC50/Dmax)'].astype('str')).toarray()
        assay_tf_idf_encoded_series = pd.Series(assay_tf_idf_encoded_array.tolist(), index=data_df.index)
        return assay_tf_idf_encoded_series

    def get_qualitative_traits_encoded_series(self, data_df):
        qualitative_traits_encoded_series = pd.Series(
            data_df[['Molecular Weight', 'Exact Mass', 'XLogP3', 'Heavy Atom Count',
                     'Ring Count', 'Hydrogen Bond Acceptor Count', 'Hydrogen Bond Donor Count', 'Rotatable Bond Count',
                     'Topological Polar Surface Area']].values.tolist())
        return qualitative_traits_encoded_series

    def get_encoded_series(self, encode_list, is_train=True):
        if is_train:
            data_df = self.training_data_df.reset_index(drop=True)
        else:
            data_df = self.testing_data_df.reset_index(drop=True)
        encoded_Series = pd.Series([[0]] * len(data_df))
        for encode in encode_list:
            if encode == "topological_fingerprints":
                topological_fingerprints_encoded_series = (
                    self.get_smiles_encoded_series(data_df, self.__get_topological_fingerprints_encode))
                encoded_Series += topological_fingerprints_encoded_series
            elif encode == "maccs":
                maccs_encoded_series = self.get_smiles_encoded_series(data_df, self.__get_maccs_encode)
                encoded_Series += maccs_encoded_series
            elif encode == "atom_pairs":
                atom_pairs_encoded_series = self.get_smiles_encoded_series(data_df, self.__get_atom_pairs_encode)
                encoded_Series += atom_pairs_encoded_series
            elif encode == "morgan_fingerprints":
                morgan_fingerprints_encoded_series = (
                    self.get_smiles_encoded_series(data_df, self.__get_morgan_fingerprints_encode))
                encoded_Series += morgan_fingerprints_encoded_series
            elif encode == "uniprot":
                uniprot_encoded_series = self.get_uniprot_id_encoded_series(data_df)
                encoded_Series += uniprot_encoded_series
            elif encode == "AAC":
                AAC_encoded_series = self.get_protein_sequence_encode_Series(
                    data_df, self.__get_single_protein_sequence_AAC_encode)
                encoded_Series += AAC_encoded_series
            elif encode == "target":
                target_encoded_series = self.get_target_encoded_series(data_df)
                encoded_Series += target_encoded_series
            elif encode == 'e3_ligase':
                e3_ligase_encoded_series = self.get_e3_ligase_encoded_series(data_df)
                encoded_Series += e3_ligase_encoded_series
            elif encode == 'assay':
                assay_tf_idf_encoded_series = self.get_assay_tf_idf_encoded_series(data_df, is_train=is_train)
                encoded_Series += assay_tf_idf_encoded_series
            elif encode == 'qualitative_trait':
                qualitative_traits_encoded_series = self.get_qualitative_traits_encoded_series(data_df)
                encoded_Series += qualitative_traits_encoded_series
            elif encode == 'targeting_ligand_and_e3_ligase_recruiter':
                targeting_ligand_and_e3_ligase_recruiter_encoded = (
                    self.get_targeting_ligand_and_e3_ligase_recruiter_encoded_series(data_df))
                encoded_Series += targeting_ligand_and_e3_ligase_recruiter_encoded
        encoded_Series = encoded_Series.apply(lambda x: x[1:])
        return encoded_Series

    def get_label_series(self):
        return self.training_data_df['Label']
