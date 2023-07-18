#!/usr/bin/env bash

echo "Attempting download of Motion Module models from Google Drive."
echo "If this fails, please download them manually from the links in the error messages/README."

gdown 1RqkQuGPaCO5sGZ6V6KZ-jUWmsRu48Kdq -O models/motion-module/ || true
gdown 1ql0g_Ys4UCz2RnokYlBjyOYPbttbIpbu -O models/motion-module/ || true

echo "Motion module download script complete."
echo "If you see errors above, please download the models manually from the links in the error messages/README."
exit 0
