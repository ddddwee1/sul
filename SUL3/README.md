# SUL3.0

## Go with KerasFlow.

##### Introduction

Incorporate tf.keras in framework.

##### Features

- Layers are now inherited from tf.keras.layers.Layer.
- More user-friendly FOR ME. Therefore, may not for others
- Although I use keras.layers, the main skeleton is aligned with sul2.

##### Note

To install, run
```
pip install tf-nightly-gpu-2.0-preview
```

Remember to change registry if using Windows.

1. Start the registry editor (regedit.exe)

2. Navigate to HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem

3. Double click LongPathsEnabled, set to 1 and click OK

4. Reboot
