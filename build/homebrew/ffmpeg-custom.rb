class FfmpegCustom < Formula
  desc "FFmpeg with stock Homebrew features plus additional codecs and libraries"
  homepage "https://ffmpeg.org/"
  url "https://ffmpeg.org/releases/ffmpeg-8.0.1.tar.xz"
  sha256 "05ee0b03119b45c0bdb4df654b96802e909e0a752f72e4fe3794f487229e5a41"
  revision 2
  license "GPL-2.0-or-later"

  depends_on "pkgconf" => :build

  # Stock Homebrew ffmpeg dependencies
  depends_on "dav1d"
  depends_on "lame"
  depends_on "libvpx"
  depends_on "openssl@3"
  depends_on "opus"
  depends_on "sdl2"
  depends_on "svt-av1"
  depends_on "x264"
  depends_on "x265"

  # Additional custom dependencies
  depends_on "aom"
  depends_on "fdk-aac"
  depends_on "freetype"
  depends_on "harfbuzz"
  depends_on "libass"
  depends_on "librist"
  depends_on "libsoxr"
  depends_on "libvmaf"
  depends_on "openjpeg"
  depends_on "srt"
  depends_on "two-lame"
  depends_on "zimg"

  uses_from_macos "bzip2"
  uses_from_macos "libxml2"
  uses_from_macos "zlib"

  on_intel do
    depends_on "nasm" => :build
  end

  on_linux do
    depends_on "alsa-lib"
    depends_on "libxcb"
    depends_on "xz"
    depends_on "zlib-ng-compat"
  end

  patch do
    url "https://gitlab.archlinux.org/archlinux/packaging/packages/ffmpeg/-/raw/5670ccd86d3b816f49ebc18cab878125eca2f81f/add-av_stream_get_first_dts-for-chromium.patch"
    sha256 "57e26caced5a1382cb639235f9555fc50e45e7bf8333f7c9ae3d49b3241d3f77"
  end

  patch do
    url "https://git.ffmpeg.org/gitweb/ffmpeg.git/patch/a5d4c398b411a00ac09d8fe3b66117222323844c"
    sha256 "1dbbc1a4cf9834b3902236abc27fefe982da03a14bcaa89fb90c7c8bd10a1664"
  end

  def install
    ENV.append "LDFLAGS", "-Wl,-ld_classic" if OS.mac? && DevelopmentTools.ld64_version.between?("1015.7", "1022.1")

    args = %W[
      --prefix=#{prefix}
      --enable-shared
      --enable-pthreads
      --enable-version3
      --enable-gpl
      --enable-nonfree

      --cc=#{ENV.cc}
      --host-cflags=#{ENV.cflags}
      --host-ldflags=#{ENV.ldflags}

      --enable-ffplay
      --enable-libsvtav1
      --enable-libopus
      --enable-libx264
      --enable-libmp3lame
      --enable-libdav1d
      --enable-libvpx
      --enable-libx265
      --enable-openssl

      --enable-libaom
      --enable-libfdk-aac
      --enable-libtwolame
      --enable-librist
      --enable-libsrt
      --enable-libass
      --enable-libfreetype
      --enable-libharfbuzz
      --enable-libsoxr
      --enable-libzimg
      --enable-libvmaf
      --enable-libopenjpeg
    ]

    if system("pkg-config", "--exists", "fribidi")
      args << "--enable-libfribidi"
    end

    if OS.mac?
      args << "--enable-videotoolbox"
      args << "--enable-audiotoolbox"
    end

    args << "--enable-neon" if Hardware::CPU.arm?

    system "./configure", *args
    system "make"
    system "make", "install"
    system "make", "alltools"

    bin.install Dir["tools/*"].select { |f| File.file?(f) && File.executable?(f) }
    pkgshare.install "tools/python" if Dir.exist?("tools/python")
  end

  test do
    output = shell_output("#{bin}/ffmpeg -version")
    assert_match "ffmpeg version", output
  end
end
