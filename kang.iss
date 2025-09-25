[Setup]
ArchitecturesInstallIn64BitMode=x64
AppId={{38F9C46D-F6F5-4819-BCEE-5A5DAB74FE95}}
AppName=AIModelCompressor
AppVersion=1.0
AppPublisher=YourName
DefaultDirName={autopf}\AIModelCompressor
DefaultGroupName=AIModelCompressor
OutputDir=.\installer
OutputBaseFilename=AIModelCompressor_Setup
Compression=lzma
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=admin

[Languages]
Name: "korean"; MessagesFile: "compiler:Languages\Korean.isl"
[Tasks]
Name: "addtopath"; Description: "Add to system PATH (recommended)"
[Files]
Source: "dist\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\AIModelCompressor"; Filename: "{app}\kang.exe"
Name: "{group}\Uninstall AIModelCompressor"; Filename: "{uninstallexe}"

[Code]
// PATH에 경로가 이미 있는지 안전하게 확인
function IsInPath(Path: string): Boolean;
var
  OrigPath: string;
begin
  if RegQueryStringValue(HKEY_LOCAL_MACHINE,
    'SYSTEM\CurrentControlSet\Control\Session Manager\Environment',
    'Path', OrigPath) then
  begin
    // 대소문자 구분 없이, 정확한 경로 매칭
    Result := Pos(';' + Uppercase(Path) + ';', ';' + Uppercase(OrigPath) + ';') > 0;
  end
  else
    Result := False;
end;

// PATH에 안전하게 추가
procedure AddToPath(NewPath: string);
var
  OrigPath: string;
  NewFullPath: string;
begin
  if not IsInPath(NewPath) then
  begin
    if RegQueryStringValue(HKEY_LOCAL_MACHINE,
      'SYSTEM\CurrentControlSet\Control\Session Manager\Environment',
      'Path', OrigPath) then
    begin
      // 기존 PATH 끝에 세미콜론과 함께 추가
      NewFullPath := OrigPath + ';' + NewPath;
    end
    else
    begin
      // PATH가 없다면 새로 생성
      NewFullPath := NewPath;
    end;
    
    // 레지스트리에 안전하게 기록
    if RegWriteExpandStringValue(HKEY_LOCAL_MACHINE,
      'SYSTEM\CurrentControlSet\Control\Session Manager\Environment',
      'Path', NewFullPath) then
    begin
      Log('Successfully added to PATH: ' + NewPath);
    end
    else
    begin
      Log('Failed to add to PATH: ' + NewPath);
    end;
  end
  else
  begin
    Log('Already in PATH: ' + NewPath);
  end;
end;

// PATH에서 안전하게 제거
procedure RemoveFromPath(PathToRemove: string);
var
  OrigPath: string;
  NewPath: string;
  P: Integer;
begin
  if RegQueryStringValue(HKEY_LOCAL_MACHINE,
    'SYSTEM\CurrentControlSet\Control\Session Manager\Environment',
    'Path', OrigPath) then
  begin
    NewPath := OrigPath;
    
    // 앞에 세미콜론이 있는 경우 제거
    P := Pos(';' + PathToRemove, NewPath);
    if P > 0 then
    begin
      Delete(NewPath, P, Length(PathToRemove) + 1);
    end
    else
    begin
      // 맨 앞에 있고 뒤에 세미콜론이 있는 경우
      P := Pos(PathToRemove + ';', NewPath);
      if P = 1 then
      begin
        Delete(NewPath, 1, Length(PathToRemove) + 1);
      end
      else
      begin
        // 경로가 전체 PATH인 경우
        if NewPath = PathToRemove then
          NewPath := '';
      end;
    end;
    
    // 레지스트리 업데이트
    if RegWriteExpandStringValue(HKEY_LOCAL_MACHINE,
      'SYSTEM\CurrentControlSet\Control\Session Manager\Environment',
      'Path', NewPath) then
    begin
      Log('Successfully removed from PATH: ' + PathToRemove);
    end
    else
    begin
      Log('Failed to remove from PATH: ' + PathToRemove);
    end;
  end;
end;

// 설치 후 PATH 추가
procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    if IsTaskSelected('addtopath') then
    begin
      AddToPath(ExpandConstant('{app}'));
    end;
  end;
end;

// 제거 시 PATH에서 삭제
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usPostUninstall then
  begin
    RemoveFromPath(ExpandConstant('{app}'));
  end;
end;