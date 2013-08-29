#include <stdio.h>
#include "nvToolsExt.h"
#include <dlfcn.h>
#include <cxxabi.h>

const char* const default_name = "Unknown";

const uint32_t colors[] = { 0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff, 0x0000ffff, 0x00ff0000, 0x00ffffff };
const int num_colors = sizeof(colors)/sizeof(uint32_t);
static int color_id = 0;

extern "C" void __cyg_profile_func_enter(void *this_fn, void *call_site) __attribute__((no_instrument_function));
extern "C" void __cyg_profile_func_exit(void *this_fn, void *call_site) __attribute__((no_instrument_function));

void rangePush(const char* const name ) __attribute__((no_instrument_function));
void rangePop() __attribute__((no_instrument_function));

extern "C" void __cyg_profile_func_enter(void *this_fn, void *call_site)
{
	Dl_info this_fn_info;
	if ( dladdr( this_fn, &this_fn_info ) )
	{
		int status = 0;
		rangePush(abi::__cxa_demangle(this_fn_info.dli_sname,0, 0, &status));
	}
	else
	{
		rangePush(default_name);
	}
} /* __cyg_profile_func_enter */

extern "C" void __cyg_profile_func_exit(void *this_fn, void *call_site)
{
	rangePop();
} /* __cyg_profile_func_enter */

void rangePush(const char* const name )
{
	nvtxEventAttributes_t eventAttrib = {0};
	eventAttrib.version = NVTX_VERSION;
	eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
	eventAttrib.colorType = NVTX_COLOR_ARGB;
	eventAttrib.color = colors[color_id];
	color_id = (color_id+1)%num_colors;
	eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
	if ( name != 0 )
	{
		eventAttrib.message.ascii = name;
	}
	else
	{
		eventAttrib.message.ascii = default_name;
	}
	nvtxRangePushEx(&eventAttrib);
}

void rangePop()
{
	nvtxRangePop();
}
