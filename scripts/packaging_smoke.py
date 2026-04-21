import platform

import senko


def main():
    system = platform.system()
    if system == "Darwin":
        diarizer = senko.Diarizer(device="auto", vad="auto", warmup=False, quiet=True)
    else:
        diarizer = senko.Diarizer(device="cpu", vad="silero", warmup=False, quiet=True)

    print(
        "PACKAGING_SMOKE_OK",
        {
            "platform": system,
            "device": diarizer.device,
            "vad_model_type": diarizer.vad_model_type,
        },
    )


if __name__ == "__main__":
    main()
