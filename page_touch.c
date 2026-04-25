#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(void)
{
    const size_t page = 4096;
    const size_t pages = 256;
    volatile uint8_t *buf;

    if (posix_memalign((void **)&buf, page, pages * page) != 0) {
        perror("posix_memalign");
        return 1;
    }

    for (size_t pass = 0; pass < 4; pass++) {
        for (size_t i = 0; i < pages; i++) {
            buf[i * page] = (uint8_t)(i + pass);
        }
    }

    uint64_t sum = 0;
    for (size_t i = 0; i < pages; i++) {
        sum += buf[i * page];
    }

    printf("sum=%lu\n", sum);
    free((void *)buf);
    return 0;
}
