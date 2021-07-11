import build_db
import query
import calcMAP
import json

if __name__ == '__main__':
    classifier = True
    modelpath = None
    #modelpath = "outputs/model_best.pth"
    #modelpath = "densenet121_pytorch_adapted.pth"
    build_db.build_db(modelpath, classifier)
    mAP = calcMAP.calc_map(modelpath, classifier)
    image_list = ["data/test/aec/B02_010257_1657_XI_14S231W_CX3340_CY3169.jpg","data/test/aec/P13_006210_2576_XN_77N271W_CX12647_CY35659.jpg",
    "data/test/ael/B08_012727_1742_XN_05S348W_CX1593_CY12594.jpg","data/test/cli/B19_017212_1809_XN_00N033W_CX4258_CY3505.jpg",
    "data/test/cli/P03_002287_2005_XI_20N072W_CX6543_CY12329.jpg","data/test/cra/B07_012260_1447_XI_35S194W_CX4750_CY4036.jpg",
    "data/test/cra/B11_014027_1420_XI_38S196W_CX6359_CY40694.jpg","data/test/fse/F07_038427_1921_XI_12N344W_CX3602_CY26346.jpg",
    "data/test/fsf/F06_038065_2069_XN_26N186W_CX4739_CY15764.jpg","data/test/fsf/P12_005635_1605_XN_19S031W_CX2511_CY7510.jpg",
    "data/test/fsg/B10_013598_1092_XN_70S355W_CX2530_CY9659.jpg","data/test/fss/F06_038140_1742_XI_05S069W_CX2186_CY13248.jpg",
    "data/test/mix/B03_010882_2041_XI_24N019W_CX1420_CY1636.jpg","data/test/rid/D18_034236_1513_XN_28S045W_CX5336_CY8224.jpg",
    "data/test/rid/P12_005575_1415_XN_38S191W_CX7162_CY35918.jpg","data/test/rou/B01_009847_1486_XI_31S197W_CX3126_CY12795.jpg",
    "data/test/sfe/B11_014000_2062_XN_26N186W_CX2066_CY2437.jpg","data/test/sfx/B01_009847_1486_XI_31S197W_CX4011_CY1016.jpg",
    "data/test/smo/B07_012490_1826_XI_02N358W_CX1369_CY939.jpg","data/test/tex/B01_009863_2303_XI_50N284W_CX11292_CY34404.jpg"]
    result_dict = {}
    result_dict["mAP"]=mAP
    #result_dict["average_distance"]=mars_dist_avg
    for imagepath in image_list:
        resultpaths, query_ap = query.query(modelpath, imagepath, classifier)
        result_dict[imagepath] = {}
        result_dict[imagepath]["queryAP"] = query_ap
        #result_dict[imagepath]["query_average_distance"] = coordinate_distance
        result_dict[imagepath]["result"] = resultpaths
        #result_dict[imagepath]["query_distances"] = distance_list

    #print(result_dict)
    with open("results/result.json", "w") as f:
        json.dump(result_dict, f, indent=4)
