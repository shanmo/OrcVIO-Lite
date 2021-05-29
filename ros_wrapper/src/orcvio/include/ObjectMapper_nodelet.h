//
// Created by xiaochen at 19-8-21.
// Nodelet for system manager.
//

#ifndef OBJECTMAPPER_NODELET_H
#define OBJECTMAPPER_NODELET_H

#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>
#include <ObjectInitNode.h>

namespace orcvio {
    class ObjectMapperNodelet : public nodelet::Nodelet {
    public:
        ObjectMapperNodelet() { return; }
        ~ObjectMapperNodelet() {
            std::cout << "in ~ObjectMapperNodelet()" << std::endl;
            return; 
            }

    private:
        virtual void onInit();
        ObjectInitPtr object_init_ptr;
    };
} // end namespace orcvio

#endif  //OBJECTMAPPER_NODELET_H
