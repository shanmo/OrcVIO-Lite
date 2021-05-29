#include <ObjectMapper_nodelet.h>

namespace orcvio {
    
void ObjectMapperNodelet::onInit() {
    
    object_init_ptr.reset(new ObjectInitNode(getPrivateNodeHandle()));
    NODELET_WARN("Initialized the object mapper Nodelet");

    return;
}

PLUGINLIB_EXPORT_CLASS(orcvio::ObjectMapperNodelet, nodelet::Nodelet);

} // end namespace orcvio

