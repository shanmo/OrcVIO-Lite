//
// Nodelet for system manager.
//

#ifndef SYSTEM_NODELET_H
#define SYSTEM_NODELET_H

#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>
#include <System.h>

namespace orcvio {
    class SystemNodelet : public nodelet::Nodelet {
    public:
        SystemNodelet() { return; }
        ~SystemNodelet() {
            // debug log
            std::cout << "in ~SystemNodelet()" << std::endl;
            return; }

    private:
        virtual void onInit();
        SystemPtr system_ptr;
    };
} // end namespace orcvio

#endif  //SYSTEM_NODELET_H
