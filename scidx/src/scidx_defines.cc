
#include <scidx_defines.h>

int scidx_versionNumber[4] = {scidx_VER_MAJOR,scidx_VER_MINOR,scidx_VER_BUILD,scidx_VER_REVISION};;
	
int scidx_dataEndianType = LITTLE_ENDIAN_DATA; //*endian type of the data read from disk
int scidx_sysEndianType = LITTLE_ENDIAN_SYSTEM; //*sysEndianType is actually set automatically.
