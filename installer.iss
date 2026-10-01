; Inno Setup script for Face-Recognition-Python 1.2.4 (one-file exe, models bundled)
; Compile with: iscc installer.iss  (requires Inno Setup 6+)
#define MyAppName "Face Recognition"
#define MyAppVersion "1.2.4"
#define MyAppPublisher "FaceRecognition"
#define MyAppExeName "FaceRecognition.exe"

[Setup]
AppId={{8E4A2B1C-6F3A-4D2E-9C1A-FACE20261}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\FaceRecognition
DefaultGroupName={#MyAppName}
OutputDir=installer-output
OutputBaseFilename=FaceRecognition_Setup_{#MyAppVersion}
Compression=lzma2/max
SolidCompression=yes
PrivilegesRequired=admin
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern
UninstallDisplayName={#MyAppName} {#MyAppVersion}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
; one-file exe already contains models + config (see face-recognition.spec)
Source: "dist\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
