#include <System_nodelet.h>

namespace orcvio {
    
void SystemNodelet::onInit() {
    system_ptr.reset(new System(getPrivateNodeHandle()));
    if (!system_ptr->initialize()) {
        ROS_ERROR("Cannot initialize System Manager...");
        return;
    }
    return;
}

PLUGINLIB_EXPORT_CLASS(orcvio::SystemNodelet, nodelet::Nodelet);

} // end namespace orcvio
