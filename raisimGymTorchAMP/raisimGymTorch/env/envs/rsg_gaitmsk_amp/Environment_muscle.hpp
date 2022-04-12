namespace raisim {
    void ENVIRONMENT::ReadMuscleModel(const std::string& FileName) {
        std::ifstream jsonfile(FileName);
        if (!jsonfile) std::cout << "Failed to open the muscle definitionfile" << std::endl;

        nlohmann::json jsondata;
        jsonfile >> jsondata;
        std::string strtemp;
        int nmuscles_temp = jsondata["nMuscles"];    // the number of muscles
        assert(nmuscles_temp == 92);

        for (int iter = 0; iter < nmuscles_temp; iter++) {    // setting muscle parameters
            Muscles_[iter].SetMuscleName(jsondata["Muscles"][iter]["Name"]);

            strtemp = jsondata["Muscles"][iter]["max_isometric_force"];
            Muscles_[iter].SetFiso(std::stod(strtemp));

            strtemp = jsondata["Muscles"][iter]["optimal_fiber_length"];
            Muscles_[iter].SetOptFiberLength(std::stod(strtemp));

            strtemp = jsondata["Muscles"][iter]["tendon_slack_length"];
            Muscles_[iter].SetTendonSlackLength(std::stod(strtemp));

            strtemp = jsondata["Muscles"][iter]["pennation_angle"];
            Muscles_[iter].SetPennationAngle(std::stod(strtemp));

            Muscles_[iter].SetNumTargetJoint(jsondata["Muscles"][iter]["nTargetJoint"]);
            if (Muscles_[iter].GetNumTargetJoint() == 1) {   // one joint muscle
                Muscles_[iter].m_strTargetJoint[0] = jsondata["Muscles"][iter]["TargetJoint"]["Joint"];
                Muscles_[iter].m_nTargetPath[0][0] = jsondata["Muscles"][iter]["TargetJoint"]["Path"][0];
                Muscles_[iter].m_nTargetPath[0][1] = jsondata["Muscles"][iter]["TargetJoint"]["Path"][1];
            }
            else {
                for (int iTargetJoint = 0; iTargetJoint < Muscles_[iter].GetNumTargetJoint(); iTargetJoint++) {
                    Muscles_[iter].m_strTargetJoint[iTargetJoint] = jsondata["Muscles"][iter]["TargetJoint"][iTargetJoint]["Joint"];
                    Muscles_[iter].m_nTargetPath[iTargetJoint][0] = jsondata["Muscles"][iter]["TargetJoint"][iTargetJoint]["Path"][0];
                    Muscles_[iter].m_nTargetPath[iTargetJoint][1] = jsondata["Muscles"][iter]["TargetJoint"][iTargetJoint]["Path"][1];
                }
            }
            int nPathPoint = jsondata["Muscles"][iter]["nPath"];
            Muscles_[iter].SetNumPathPoint(nPathPoint); assert(nPathPoint < 9);

            double temp_double[3];
            std::string mystring;
            for (int ipath = 0; ipath < nPathPoint; ipath++) {
                std::stringstream ss2;
                ss2 << ipath + 1;
                mystring = "path" + ss2.str();

                temp_double[0] = jsondata["Muscles"][iter]["PathSet"][mystring]["locationx"];
                temp_double[1] = jsondata["Muscles"][iter]["PathSet"][mystring]["locationy"];
                temp_double[2] = jsondata["Muscles"][iter]["PathSet"][mystring]["locationz"];

                Muscles_[iter].SetPathPosLocal(ipath, temp_double[0], temp_double[1], temp_double[2]);
                Muscles_[iter].SetPathLink(ipath, jsondata["Muscles"][iter]["PathSet"][mystring]["body"]);
                Muscles_[iter].SetPathType(ipath, jsondata["Muscles"][iter]["PathSet"][mystring]["type"]);
                if (Muscles_[iter].GetPathType(ipath) == "Conditional") {
                    Muscles_[iter].Coordinate[ipath] = jsondata["Muscles"][iter]["PathSet"][mystring]["coordinate"];
                    Muscles_[iter].Range[ipath] = jsondata["Muscles"][iter]["PathSet"][mystring]["range"];
                }
                if (Muscles_[iter].GetPathType(ipath) == "Moving") {
                    Muscles_[iter].CoordinateX[ipath] = jsondata["Muscles"][iter]["PathSet"][mystring]["x_coordinate"];
                    Muscles_[iter].CoordinateY[ipath] = jsondata["Muscles"][iter]["PathSet"][mystring]["y_coordinate"];
                    Muscles_[iter].CoordinateZ[ipath] = jsondata["Muscles"][iter]["PathSet"][mystring]["z_coordinate"];
                    Muscles_[iter].x_x[ipath] = jsondata["Muscles"][iter]["PathSet"][mystring]["x_location"]["x"];
                    Muscles_[iter].x_y[ipath] = jsondata["Muscles"][iter]["PathSet"][mystring]["x_location"]["y"];
                    Muscles_[iter].y_x[ipath] = jsondata["Muscles"][iter]["PathSet"][mystring]["y_location"]["x"];
                    Muscles_[iter].y_y[ipath] = jsondata["Muscles"][iter]["PathSet"][mystring]["y_location"]["y"];
                    Muscles_[iter].z_x[ipath] = jsondata["Muscles"][iter]["PathSet"][mystring]["z_location"]["x"];
                    Muscles_[iter].z_y[ipath] = jsondata["Muscles"][iter]["PathSet"][mystring]["z_location"]["y"];
                    double d = 0.0;
                    std::stringstream sx(Muscles_[iter].x_x[ipath]);
                    while (sx >> d)
                        Muscles_[iter].vx.push_back(d);
                    std::stringstream sy(Muscles_[iter].x_y[ipath]);
                    while (sy >> d)
                        Muscles_[iter].vy.push_back(d);

                    std::stringstream sx2(Muscles_[iter].y_x[ipath]);
                    while (sx2 >> d)
                        Muscles_[iter].vx2.push_back(d);
                    std::stringstream sy2(Muscles_[iter].y_y[ipath]);
                    while (sy2 >> d)
                        Muscles_[iter].vy2.push_back(d);

                    std::vector<double> b_temp, c_temp, d_temp;
                    b_temp.clear();c_temp.clear();d_temp.clear();
                    MuscleModel::calcCoefficients(Muscles_[iter].vx, Muscles_[iter].vy, b_temp, c_temp, d_temp);
                    for (int j = 0; j < Muscles_[iter].vx.size(); j++) {
                        Muscles_[iter].x_b.push_back(b_temp.at(j));
                        Muscles_[iter].x_c.push_back(c_temp.at(j));
                        Muscles_[iter].x_d.push_back(d_temp.at(j));
                    }
                    b_temp.clear();c_temp.clear();d_temp.clear();
                    MuscleModel::calcCoefficients(Muscles_[iter].vx2, Muscles_[iter].vy2, b_temp, c_temp, d_temp);
                    for (int j = 0; j < Muscles_[iter].vx2.size(); j++) {
                        Muscles_[iter].y_b.push_back(b_temp.at(j));
                        Muscles_[iter].y_c.push_back(c_temp.at(j));
                        Muscles_[iter].y_d.push_back(d_temp.at(j));
                    }
                }
            } // for ipath
        }
    }


    void ENVIRONMENT::GetScaleFactor(std::string file_path) {
        std::ifstream f(file_path);
        std::stringstream ss;
        ss << f.rdbuf();
        std::string urdf_str = ss.str();
        //auto urdf_str_size = urdf_str.size();

        scale_link_[0] = "pelvis";
        scale_link_[1] = "femur_r";
        scale_link_[2] = "tibia_r";
        scale_link_[3] = "talus_r";
        scale_link_[4] = "calcn_r";
        scale_link_[5] = "toes_r";
        scale_link_[6] = "femur_l";
        scale_link_[7] = "tibia_l";
        scale_link_[8] = "talus_l";
        scale_link_[9] = "calcn_l";
        scale_link_[10] = "toes_l";
        scale_link_[11] = "torso";
        scale_link_[12] = "humerus_r";
        scale_link_[13] = "ulna_r";
        scale_link_[14] = "radius_r";
        scale_link_[15] = "hand_r";
        scale_link_[16] = "humerus_l";
        scale_link_[17] = "ulna_l";
        scale_link_[18] = "radius_l";
        scale_link_[19] = "hand_l";

        std::size_t idx_link[20], idx_scale1[20], idx_scale2[20];

        for (int iter = 0; iter < 20; iter++) {
            idx_link[iter] = urdf_str.find(scale_link_[iter]);
            /// print link name
            //std::cout << temp.substr(idx_link[iter], scale_link_[iter].length()) << std::endl;

            idx_scale1[iter] = urdf_str.find("scale=", idx_link[iter]) + 7;
            idx_scale2[iter] = urdf_str.find(" ", idx_scale1[iter]);
            std::size_t scale_length = (idx_scale2[iter]) - (idx_scale1[iter]);
            //std::cout << "scale: " << temp.substr(idx_scale1[iter], scale_length) << std::endl;

            scale_value_[iter] = std::stod(urdf_str.substr(idx_scale1[iter], scale_length));
            //std::cout << "scale: " << scale_value_[iter] << std::endl;
        }
    }


    void ENVIRONMENT::ReadMuscleInitialLength(std::string FileName, double *L0) {
        std::ifstream jsonfile(FileName);
        nlohmann::json jsondata;
        jsonfile >> jsondata;
        std::string strtemp;

        for (int iter = 0; iter < 92; iter++) {    // setting muscle parameters
            L0[iter] = jsondata[Muscles_[iter].GetMuscleName()][0];
        }
    }


    void ENVIRONMENT::update_seg_txf() {
        raisim::Vec<3> TrsVec;
        raisim::Mat<3, 3> RotMat{};

        /// rotation and translation of pelvis
        anymal_->getBasePosition(TrsVec);
        anymal_->getBaseOrientation(RotMat);
        muscle_trs_[0] = TrsVec;
        muscle_rot_[0] = RotMat;

        /// rotation and translation of other segments
        auto str_temp = anymal_->getBodyNames();
        for (int i = 1; i < anymal_->getNumberOfJoints(); i++) {
            anymal_->getLink(str_temp.at(i)).getPose(TrsVec, RotMat);
            muscle_trs_[i] = TrsVec;
            muscle_rot_[i] = RotMat;
        }
    }
}