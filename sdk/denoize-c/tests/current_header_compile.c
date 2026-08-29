#include "denoize.h"

#include <stddef.h>

_Static_assert(DENOIZE_ABI_VERSION_V1 == 1, "unexpected ABI version");
_Static_assert(sizeof(void *) != 8 || sizeof(denoize_options_v1) == 96,
               "64-bit options layout changed");

static denoize_status (*const create_fn)(
    const denoize_options_v1 *, denoize_processor **, denoize_cancel_token **,
    denoize_diagnostic_v1 *) = denoize_processor_create_v1;

int main(void) {
    (void)create_fn;
    return 0;
}
