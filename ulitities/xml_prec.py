# coding:utf-8

import os
import sys
import xml.etree.ElementTree as ET
import xml.dom.minidom as Document
import xml.dom


# def dict_to_xml(input_dict,root_tag,node_tag):
#     """ 定义根节点root_tag，定义第二层节点node_tag
#     第三层中将字典中键值对对应参数名和值
#        return: xml的tree结构 """
#     root_name = ET.Element(root_tag)
#     for (k, v) in input_dict.items():
#         node_name = ET.SubElement(root_name, node_tag)
#         for (key, val) in sorted(v.items(), key=lambda e:e[0], reverse=True):
#             key = ET.SubElement(node_name, key)
#             key.text = val
#     return root_name


# doc = Document()
# root = doc.createElement('root')




def generate_xml_from_dict(input_dict, xml_file):


    impl = xml.dom.getDOMImplementation()
    dom = impl.createDocument(None, 'root', None)
    root = dom.documentElement
    instance = dom.createElement('instance')
    root.appendChild(instance)

    for (key, value) in sorted(input_dict.items()):
        nameE = dom.createElement(key)
        nameT = dom.createTextNode(str(value))
        nameE.appendChild(nameT)
        instance.appendChild(nameE)

    with open(xml_file, 'w', encoding='utf-8') as tp:
        dom.writexml(tp, addindent='    ', newl='\n', encoding='utf-8')


def parse_xml_to_dict(xml_file):

    if not os.path.isfile(xml_file):
        print("input file is not existed!:{}".format(xml_file))
        sys.exit(-1)

    tree = ET.parse(xml_file)
    root = tree.getroot()
    dict_new = {}
    for key, value in enumerate(root):
        dict_init = {}
        list_init = []
        for item in value:
            list_init.append([item.tag, item.text])
            for lists in list_init:
                dict_init[lists[0]] = lists[1]
        dict_new[key] = dict_init
    return dict_new





if __name__=='__main__':
    # imgStretch_dict = {'input_dir': '', 'output_dir': '', 'NoData': 65535, 'OutBits': '16bits',
    #                    'StretchRange': '1024',
    #                    'CutValue': '100'}
    save_file = 'imgstretch.xml'
    # generate_xml_from_dict(imgStretch_dict, save_file)
    dict_one = parse_xml_to_dict(save_file)
    print(dict_one[0])