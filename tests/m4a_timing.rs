use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use denoize::{
    decode_file, inspect_audio_stream_session, probe_file, AudioCodec, AudioFormat,
    AudioInputSession, AudioStreamReader, ChannelLayout, DecodeLimits, DecodedPcm,
};
use sha2::{Digest, Sha256};

// These synthetic fixtures were generated twice and compared byte-for-byte with
// FFmpeg 6.1.1-3ubuntu5+esm10 (libavcodec 60.31.102, libavformat 60.16.100).
// The edit-list fixtures were produced with:
//
//   ffmpeg -hide_banner -loglevel error -f lavfi \
//     -i 'sine=frequency=1000:sample_rate=44100:duration=0.1' -map 0:a:0 \
//     -ac 1 -c:a aac -profile:a aac_low -b:a 64k -movflags +faststart \
//     -metadata title='Synthetic M4A edit-list fixture' \
//     -metadata comment='Generated waveform; no source recording' \
//     -y candidate_mono44100_faststart_meta_a.m4a
//
//   ffmpeg -hide_banner -loglevel error -f lavfi \
//     -i 'aevalsrc=0.2*sin(2*PI*440*t)|0.15*sin(2*PI*733*t):s=48000' \
//     -af 'atrim=end_sample=1009' -map 0:a:0 -ac 2 -c:a aac \
//     -profile:a aac_low -b:a 96k -movie_timescale 1001 \
//     -y candidate_stereo48000_n1009_ts1001_a.m4a
//
// The leading-empty fixture was likewise generated twice and compared with:
//
//   ffmpeg -hide_banner -loglevel error -itsoffset 0.25 -f lavfi \
//     -i 'sine=frequency=1000:sample_rate=48000:duration=0.1' -map 0:a:0 \
//     -ac 1 -c:a aac -profile:a aac_low -b:a 64k \
//     -y leading-empty-250ms.m4a
//
// The oversized version-1 edit-list fixture was generated twice and compared
// byte-for-byte with:
//
//   ffmpeg -hide_banner -loglevel error -itsoffset 5000000 -f lavfi \
//     -i 'sine=frequency=1000:sample_rate=48000:duration=0.1' -map 0:a:0 \
//     -ac 1 -c:a aac -profile:a aac_low -b:a 64k -movie_timescale 1000 \
//     -y oversized-v1-edit-list.m4a
//
// The mono, stereo, and leading-empty no-edit controls repeat their corresponding
// commands with `-use_editlist 0` immediately before `-y`. Thus each pair has the
// same encoded AAC packets, while only the candidate carries the presentation
// edit list under test. The byte lengths and SHA-256 digests below pin the exact
// audited artifacts used by these tests.
const MONO_EDIT_LIST: &str = concat!(
    "AAAAHGZ0eXBNNEEgAAACAE00QSBpc29taXNvMgAAA4ltb292AAAAbG12aGQAAAAAAAAAAAAAAAAAAAPoAAAAZAABAAABAAAA",
    "AAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAC",
    "AAACPXRyYWsAAABcdGtoZAAAAAMAAAAAAAAAAAAAAAEAAAAAAAAAZAAAAAAAAAAAAAAAAQEAAAAAAQAAAAAAAAAAAAAAAAAA",
    "AAEAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAACRlZHRzAAAAHGVsc3QAAAAAAAAAAQAAAGQAAAQAAAEAAAAAAbVtZGlh",
    "AAAAIG1kaGQAAAAAAAAAAAAAAAAAAKxEAAAVOlXEAAAAAAAtaGRscgAAAAAAAAAAc291bgAAAAAAAAAAAAAAAFNvdW5kSGFu",
    "ZGxlcgAAAAFgbWluZgAAABBzbWhkAAAAAAAAAAAAAAAkZGluZgAAABxkcmVmAAAAAAAAAAEAAAAMdXJsIAAAAAEAAAEkc3Ri",
    "bAAAAGpzdHNkAAAAAAAAAAEAAABabXA0YQAAAAAAAAABAAAAAAAAAAAAAQAQAAAAAKxEAAAAAAA2ZXNkcwAAAAADgICAJQAB",
    "AASAgIAXQBUAAAAAAQ3XAAEN1wWAgIAFEghW5QAGgICAAQIAAAAgc3R0cwAAAAAAAAACAAAABQAABAAAAAABAAABOgAAABxz",
    "dHNjAAAAAAAAAAEAAAABAAAABgAAAAEAAAAsc3RzegAAAAAAAAAAAAAABgAAAQ8AAADzAAAAlQAAAJAAAAD8AAAABQAAABRz",
    "dGNvAAAAAAAAAAEAAAO1AAAAGnNncGQBAAAAcm9sbAAAAAIAAAAB//8AAAAcc2JncAAAAAByb2xsAAAAAQAAAAYAAAABAAAA",
    "2HVkdGEAAADQbWV0YQAAAAAAAAAhaGRscgAAAAAAAAAAbWRpcmFwcGwAAAAAAAAAAAAAAACjaWxzdAAAADepbmFtAAAAL2Rh",
    "dGEAAAABAAAAAFN5bnRoZXRpYyBNNEEgZWRpdC1saXN0IGZpeHR1cmUAAAAlqXRvbwAAAB1kYXRhAAAAAQAAAABMYXZmNjAu",
    "MTYuMTAwAAAAP6ljbXQAAAA3ZGF0YQAAAAEAAAAAR2VuZXJhdGVkIHdhdmVmb3JtOyBubyBzb3VyY2UgcmVjb3JkaW5nAAAA",
    "CGZyZWUAAAQwbWRhdN4CAExhdmM2MC4zMS4xMDIAAnCsW6oErSO+vr+v6r71+evjjelpUmSSSdxEB0ICZKMKmS4tmEYWeqZp",
    "7CdxcWz1TvJX0LayLyvkwUuG4JqDTrVDubim5vYebuNdlWW5TtaynKsbirjPVmextisNana1Ox0bHRsdGx0bHRscpVKVSlUp",
    "VKVSlUpVKVSlUpVKVRJRJRJRJRJRJRJRJRJRJRJRJRJRJRJRJRJRJRJRJRJRJRJRJRJRJRJRJRJRJRJRJRJRJRJRJRJRJRJR",
    "JRJRJRJRJUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUXABCpja",
    "003WYjEdZipCV1Npf/X/Pm9PfV3q8/+v/l+tknF6/t/+3/P4uGtXP7//t/7fqi9XdjMX4Fv37Q8zs7Owzd5LEv/qhw/fe8Z7",
    "F+qRkQ2hfCqfFh2k/KoypLDeOFL01s6wsfy/lr15n8v5a9Z/L+Utcw/l8P5a9cx/KOHXMPh8PhLxzlIxnzxO2fXsRtISPfkd",
    "/qRtA/mlVIu9x6222tVev2s6paefeOyUSJlpEUsoMDAwNKoGlBgaVTgwMiBgY3Li4MDAwMDUCIfKbj8vlOQA2RVCoRbIj0Vj",
    "AnLKSW8oovo2pga8OvDiV4GBiQMbr8ABAvYpFnJOoIOhE2hEb7Nzfn5yvnz9+r/bU1qruP+1xc/FrjJ1dQUXlTZs2NHitZ1q",
    "1yl9GkoyxLcX16Ms/Vs0mRd0h7xxWK9gzCuYsSzFmKmqacQ7D7+8I+Pgwz++gfHwYZ/eEZPhsO+5AH+GzCzOH8aGCyvHklgs",
    "onOXiKVjW7ciFkJxmlCBiz0QuKIXnOtO0b3q4AEIVi20phbNznGPUzf9v5+uL4kkyWp+WlrRaSSkHZex3WfQOzSHZGYMwUwq",
    "I3fifG/G/a+x+yxy1eVkTMTC57nv/z+/+v8X/09vXpM46p1TjH1HGHx3Z+P9X6v1fK391IWTaNoyiwKAWdnxeLxceWgIwaNG",
    "igg4c23bx7cs8Rg0aePSUYyy8vVlAKueecohwAEEmbLNROLMZELvabWsPX/8X/Xzq41q7/j/+1/5/fUF6r+v/b/b8TUHOrCK",
    "Z3FrWgh+skgpqN7I8fAVbQF+ortdVPhcKujVRBa0pSa2GX5Nob/jfJpZQYGBgZTopQYGxxc3LKDAwMqVzZs2DAwNi8puVkGN",
    "m8VO4NLLEDAyJF27aTMsz4uJ5AF+hX6EWXFwaXDke/M/LsuzsNfM2OVq7As2TSotYcEMCoYE2UkUoVLVBCkiDA1699J1SDAw",
    "69ZZok5N/P277mzY/n7fcufn6QDXr8f8iK4RItCcxWtAbjINNZ6ZYRMxLtQSEDIIK2BsvUGotC5qxG5qV5BHfAEYgbRw",
);

const MONO_NO_EDIT: &str = concat!(
    "AAAAHGZ0eXBNNEEgAAACAE00QSBpc29taXNvMgAAA2Vtb292AAAAbG12aGQAAAAAAAAAAAAAAAAAAAPoAAAAfAABAAABAAAA",
    "AAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAC",
    "AAACGXRyYWsAAABcdGtoZAAAAAMAAAAAAAAAAAAAAAEAAAAAAAAAfAAAAAAAAAAAAAAAAQEAAAAAAQAAAAAAAAAAAAAAAAAA",
    "AAEAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAbVtZGlhAAAAIG1kaGQAAAAAAAAAAAAAAAAAAKxEAAAVOlXEAAAAAAAt",
    "aGRscgAAAAAAAAAAc291bgAAAAAAAAAAAAAAAFNvdW5kSGFuZGxlcgAAAAFgbWluZgAAABBzbWhkAAAAAAAAAAAAAAAkZGlu",
    "ZgAAABxkcmVmAAAAAAAAAAEAAAAMdXJsIAAAAAEAAAEkc3RibAAAAGpzdHNkAAAAAAAAAAEAAABabXA0YQAAAAAAAAABAAAA",
    "AAAAAAAAAQAQAAAAAKxEAAAAAAA2ZXNkcwAAAAADgICAJQABAASAgIAXQBUAAAAAAQ3XAAEN1wWAgIAFEghW5QAGgICAAQIA",
    "AAAgc3R0cwAAAAAAAAACAAAABQAABAAAAAABAAABOgAAABxzdHNjAAAAAAAAAAEAAAABAAAABgAAAAEAAAAsc3RzegAAAAAA",
    "AAAAAAAABgAAAQ8AAADzAAAAlQAAAJAAAAD8AAAABQAAABRzdGNvAAAAAAAAAAEAAAORAAAAGnNncGQBAAAAcm9sbAAAAAIA",
    "AAAB//8AAAAcc2JncAAAAAByb2xsAAAAAQAAAAYAAAABAAAA2HVkdGEAAADQbWV0YQAAAAAAAAAhaGRscgAAAAAAAAAAbWRp",
    "cmFwcGwAAAAAAAAAAAAAAACjaWxzdAAAADepbmFtAAAAL2RhdGEAAAABAAAAAFN5bnRoZXRpYyBNNEEgZWRpdC1saXN0IGZp",
    "eHR1cmUAAAAlqXRvbwAAAB1kYXRhAAAAAQAAAABMYXZmNjAuMTYuMTAwAAAAP6ljbXQAAAA3ZGF0YQAAAAEAAAAAR2VuZXJh",
    "dGVkIHdhdmVmb3JtOyBubyBzb3VyY2UgcmVjb3JkaW5nAAAACGZyZWUAAAQwbWRhdN4CAExhdmM2MC4zMS4xMDIAAnCsW6oE",
    "rSO+vr+v6r71+evjjelpUmSSSdxEB0ICZKMKmS4tmEYWeqZp7CdxcWz1TvJX0LayLyvkwUuG4JqDTrVDubim5vYebuNdlWW5",
    "TtaynKsbirjPVmextisNana1Ox0bHRsdGx0bHRscpVKVSlUpVKVSlUpVKVSlUpVKVRJRJRJRJRJRJRJRJRJRJRJRJRJRJRJR",
    "JRJRJRJRJRJRJRJRJRJRJRJRJRJRJRJRJRJRJRJRJRJRJRJRJRJRJRJRJUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU",
    "UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUXABCpja003WYjEdZipCV1Npf/X/Pm9PfV3q8/+v/l+tknF6/t/+3/P4",
    "uGtXP7//t/7fqi9XdjMX4Fv37Q8zs7Owzd5LEv/qhw/fe8Z7F+qRkQ2hfCqfFh2k/KoypLDeOFL01s6wsfy/lr15n8v5a9Z/",
    "L+Utcw/l8P5a9cx/KOHXMPh8PhLxzlIxnzxO2fXsRtISPfkd/qRtA/mlVIu9x6222tVev2s6paefeOyUSJlpEUsoMDAwNKoG",
    "lBgaVTgwMiBgY3Li4MDAwMDUCIfKbj8vlOQA2RVCoRbIj0VjAnLKSW8oovo2pga8OvDiV4GBiQMbr8ABAvYpFnJOoIOhE2hE",
    "b7Nzfn5yvnz9+r/bU1qruP+1xc/FrjJ1dQUXlTZs2NHitZ1q1yl9GkoyxLcX16Ms/Vs0mRd0h7xxWK9gzCuYsSzFmKmqacQ7",
    "D7+8I+Pgwz++gfHwYZ/eEZPhsO+5AH+GzCzOH8aGCyvHklgsonOXiKVjW7ciFkJxmlCBiz0QuKIXnOtO0b3q4AEIVi20phbN",
    "znGPUzf9v5+uL4kkyWp+WlrRaSSkHZex3WfQOzSHZGYMwUwqI3fifG/G/a+x+yxy1eVkTMTC57nv/z+/+v8X/09vXpM46p1T",
    "jH1HGHx3Z+P9X6v1fK391IWTaNoyiwKAWdnxeLxceWgIwaNGigg4c23bx7cs8Rg0aePSUYyy8vVlAKueecohwAEEmbLNROLM",
    "ZELvabWsPX/8X/Xzq41q7/j/+1/5/fUF6r+v/b/b8TUHOrCKZ3FrWgh+skgpqN7I8fAVbQF+ortdVPhcKujVRBa0pSa2GX5N",
    "ob/jfJpZQYGBgZTopQYGxxc3LKDAwMqVzZs2DAwNi8puVkGNm8VO4NLLEDAyJF27aTMsz4uJ5AF+hX6EWXFwaXDke/M/Lsuz",
    "sNfM2OVq7As2TSotYcEMCoYE2UkUoVLVBCkiDA1699J1SDAw69ZZok5N/P277mzY/n7fcufn6QDXr8f8iK4RItCcxWtAbjIN",
    "NZ6ZYRMxLtQSEDIIK2BsvUGotC5qxG5qV5BHfAEYgbRw",
);

const STEREO_EDIT_LIST: &str = concat!(
    "AAAAHGZ0eXBNNEEgAAACAE00QSBpc29taXNvMgAAAAhmcmVlAAAB0m1kYXTeAgBMYXZjNjAuMzEuMTAyAEIzlAD/8AAUsHaJ",
    "C0KD05B0JM9dcZqXfV3Lk5jpu3Hqq1Nydb1QaZEFqi1UyvsdrOmtOzs7Ozs7O2GMk2GMk2GOE2GMk2GOGOGOGOGOGOGOGLs7",
    "Ozs7Ozs7OztJNJNJNJNJNJNJNIztFFFFFFFFFFEDAwMDAwMDAwNLLLLLLLLLLLLKiKKKKKKKdhgs1736lZz4v21xq7lycx0e",
    "EiSyS6BVVIgtMtMtUWqLVGV9itKmlMtElMtMlYRMeps2R2bIgAAOIRoLE/ty3wTbGTuld+OfWusuaXJN9N9L7nXPU78oPsgF",
    "YlSClTbvaiMGBr169evBgYGHppppppgkJCbt27du3QMDMyyyygwMDA169evXhgYGTTTTTBISEhN27du3QMDAwOWWWUGBgYGB",
    "r169evDD00000wSEhISE3bt26BgYGBmZZZQYGBgYGJXr14MDAwMPTTTJQkJoUWLfCRPCRdKnisy/nXtpcksst3IiIPAAE9sM",
    "u3boN965lLKoAP5fy2RDX/b+0wNWrU379+/fRRRQYc7zzzzzwACIoooiB56KKKKAEvPPPMAEUUUUURA888884AXRRRQYcAAA",
    "AwNtb292AAAAbG12aGQAAAAAAAAAAAAAAAAAAAPpAAAAFgABAAABAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAEAAAAA",
    "AAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACAAACLXRyYWsAAABcdGtoZAAAAAMAAAAAAAAAAAAA",
    "AAEAAAAAAAAAFgAAAAAAAAAAAAAAAQEAAAAAAQAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAA",
    "ACRlZHRzAAAAHGVsc3QAAAAAAAAAAQAAABUAAAQAAAEAAAAAAaVtZGlhAAAAIG1kaGQAAAAAAAAAAAAAAAAAALuAAAAH8VXE",
    "AAAAAAAtaGRscgAAAAAAAAAAc291bgAAAAAAAAAAAAAAAFNvdW5kSGFuZGxlcgAAAAFQbWluZgAAABBzbWhkAAAAAAAAAAAA",
    "AAAkZGluZgAAABxkcmVmAAAAAAAAAAEAAAAMdXJsIAAAAAEAAAEUc3RibAAAAGpzdHNkAAAAAAAAAAEAAABabXA0YQAAAAAA",
    "AAABAAAAAAAAAAAAAgAQAAAAALuAAAAAAAA2ZXNkcwAAAAADgICAJQABAASAgIAXQBUAAAAAAXcAAAFR7AWAgIAFEZBW5QAG",
    "gICAAQIAAAAgc3R0cwAAAAAAAAACAAAAAQAABAAAAAABAAAD8QAAABxzdHNjAAAAAAAAAAEAAAABAAAAAgAAAAEAAAAcc3Rz",
    "egAAAAAAAAAAAAAAAgAAANMAAAD3AAAAFHN0Y28AAAAAAAAAAQAAACwAAAAac2dwZAEAAAByb2xsAAAAAgAAAAH//wAAABxz",
    "YmdwAAAAAHJvbGwAAAABAAAAAgAAAAEAAABidWR0YQAAAFptZXRhAAAAAAAAACFoZGxyAAAAAAAAAABtZGlyYXBwbAAAAAAA",
    "AAAAAAAAAC1pbHN0AAAAJal0b28AAAAdZGF0YQAAAAEAAAAATGF2ZjYwLjE2LjEwMA==",
);

const STEREO_NO_EDIT: &str = concat!(
    "AAAAHGZ0eXBNNEEgAAACAE00QSBpc29taXNvMgAAAAhmcmVlAAAB0m1kYXTeAgBMYXZjNjAuMzEuMTAyAEIzlAD/8AAUsHaJ",
    "C0KD05B0JM9dcZqXfV3Lk5jpu3Hqq1Nydb1QaZEFqi1UyvsdrOmtOzs7Ozs7O2GMk2GMk2GOE2GMk2GOGOGOGOGOGOGOGLs7",
    "Ozs7Ozs7OztJNJNJNJNJNJNJNIztFFFFFFFFFFEDAwMDAwMDAwNLLLLLLLLLLLLKiKKKKKKKdhgs1736lZz4v21xq7lycx0e",
    "EiSyS6BVVIgtMtMtUWqLVGV9itKmlMtElMtMlYRMeps2R2bIgAAOIRoLE/ty3wTbGTuld+OfWusuaXJN9N9L7nXPU78oPsgF",
    "YlSClTbvaiMGBr169evBgYGHppppppgkJCbt27du3QMDMyyyygwMDA169evXhgYGTTTTTBISEhN27du3QMDAwOWWWUGBgYGB",
    "r169evDD00000wSEhISE3bt26BgYGBmZZZQYGBgYGJXr14MDAwMPTTTJQkJoUWLfCRPCRdKnisy/nXtpcksst3IiIPAAE9sM",
    "u3boN965lLKoAP5fy2RDX/b+0wNWrU379+/fRRRQYc7zzzzzwACIoooiB56KKKKAEvPPPMAEUUUUURA888884AXRRRQYcAAA",
    "At9tb292AAAAbG12aGQAAAAAAAAAAAAAAAAAAAPpAAAAKwABAAABAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAEAAAAA",
    "AAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACAAACCXRyYWsAAABcdGtoZAAAAAMAAAAAAAAAAAAA",
    "AAEAAAAAAAAAKwAAAAAAAAAAAAAAAQEAAAAAAQAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAA",
    "AaVtZGlhAAAAIG1kaGQAAAAAAAAAAAAAAAAAALuAAAAH8VXEAAAAAAAtaGRscgAAAAAAAAAAc291bgAAAAAAAAAAAAAAAFNv",
    "dW5kSGFuZGxlcgAAAAFQbWluZgAAABBzbWhkAAAAAAAAAAAAAAAkZGluZgAAABxkcmVmAAAAAAAAAAEAAAAMdXJsIAAAAAEA",
    "AAEUc3RibAAAAGpzdHNkAAAAAAAAAAEAAABabXA0YQAAAAAAAAABAAAAAAAAAAAAAgAQAAAAALuAAAAAAAA2ZXNkcwAAAAAD",
    "gICAJQABAASAgIAXQBUAAAAAAXcAAAFR7AWAgIAFEZBW5QAGgICAAQIAAAAgc3R0cwAAAAAAAAACAAAAAQAABAAAAAABAAAD",
    "8QAAABxzdHNjAAAAAAAAAAEAAAABAAAAAgAAAAEAAAAcc3RzegAAAAAAAAAAAAAAAgAAANMAAAD3AAAAFHN0Y28AAAAAAAAA",
    "AQAAACwAAAAac2dwZAEAAAByb2xsAAAAAgAAAAH//wAAABxzYmdwAAAAAHJvbGwAAAABAAAAAgAAAAEAAABidWR0YQAAAFpt",
    "ZXRhAAAAAAAAACFoZGxyAAAAAAAAAABtZGlyYXBwbAAAAAAAAAAAAAAAAC1pbHN0AAAAJal0b28AAAAdZGF0YQAAAAEAAAAA",
    "TGF2ZjYwLjE2LjEwMA==",
);

const LEADING_EMPTY_EDIT_LIST: &str = concat!(
    "AAAAHGZ0eXBNNEEgAAACAE00QSBpc29taXNvMgAAAAhmcmVlAAAERm1kYXTeAgBMYXZjNjAuMzEuMTAyAAJgqlIstKYqL0JB",
    "0KuPn2VU57+vMsvEk3JJNxCB2SBs1bNcqzk0ZxVhxWNs2Ns2U5i3DZXTOciAB/oYz3Da341HdQ6+1ri3VeG4HXuO8NtO3Y3F",
    "Y3FXGerMdicdYa1O1pq2UtlKpSqUqlKpSqUqlKpSqUqlJpSaUqlKoyqMmpJpGkZ2dpJnmdnZ2kaRnZ2dnaRpGdnZ2dnmdnZ2",
    "dnZ20sDAwMDIpZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZYoo",
    "ooooouABGpja2j3WYiYTZiJhN1MJf9P864jy1etR//F/3/Wya1rP4//i/7/rZNa1n2//i/5+9l641yIy7Rkc6tChISEtPnru",
    "jZMRn66SknsnUzcT6apwe9fs5G2qbz+GCqZ/yy+Ne/5ITLLevBITKqxwkJlle87hISEqld3cqEhMsqr2bhISEwEAD567o2TE",
    "Z+ukpJ7J1M3E+mqcHvX7ORtqm8/hgqmf8svjXv+SEyy3rwSEyqscJCZZXvO4SEhKpXd3KhITLKq9m4SEhMBABeNjlcK98L78",
    "8vC//LLLLLLLLKDSyopZYDgA9vUtlFYZD2AhoIhf2+3nit7yf2/63rSpbdyDSWkkkkXVf/f/6DVuTRMLSKKlSrIjFGUZYlT2",
    "JW1mLsroN3y0Usb6l693Vxd+261lUcuFkZ3acpsLUkZk2bNjRSFXhhhhIJ0444zMhe/voHx8Zh7++iABEy1atWqklAZk2bNj",
    "QQLnAPQ1GJaqMg9DQdHYdEIQC4QE4QDIgC4Tjx96xv/0/1///e1pWl/aR5Xcl3X8fFc8A7O2GLs7YNBYyjGnsR3FiOkjdE5Z",
    "wASQqpktlhd3GuxtI5i412NmHEtp2rOcVnOKxNmnZ6RbYmzWWenZ6dnpGSatp2estHrdiBgYGBgYGBgYGBgYEIEIEIE7tvJm",
    "gs9WLjDTDQAACIAAR3vOAZ98AAAA1/wtgAAAI/E4QcABEjUttGYqC1xB0Lmp35TPf+v/zkzWRktNx+JJIkQ+g+Io3SchyJij",
    "hz+63r1i8OZ/ydtZMF4J6nmPnrcmChezCEVn2kmQ1Ti1Z+Qq3cfudF6poFHla6ybZA7XTvE4TJx8ycWM48poodty7M3ynPOq",
    "7Js5bcqLM3ynPMbp2slLJLJXynMwHwUT4BHSMB8GGMfhJXd2cAEkNS30lg6Fg68TFNvX4cXmdKripj7u4kkREBhvdL84y5uk",
    "v7jsnDu6o37d5trz7lsrPP1GqMnJycIkMX76NsFDdobI8629p+xY1393lD9w9U192tSWR4yozqOmzq3JcaV5zjYLaoau2ExZ",
    "n5LOrWUcG1UjNnMjYTE8/JZ2UybBtVIzZyxagNlwrVwyVRhJTilch41yHjNXgNMxYcAAAAMfbW9vdgAAAGxtdmhkAAAAAAAA",
    "AAAAAAAAAAAD6AAAAV4AAQAAAQAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAA",
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAgAAAkl0cmFrAAAAXHRraGQAAAADAAAAAAAAAAAAAAABAAAAAAAAAV4AAAAAAAAAAAAA",
    "AAEBAAAAAAEAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAwZWR0cwAAAChlbHN0AAAAAAAA",
    "AAIAAADk/////wABAAAAAAB6AAAAAAABAAAAAAG1bWRpYQAAACBtZGhkAAAAAAAAAAAAAAAAAAC7gAAAFsBVxAAAAAAALWhk",
    "bHIAAAAAAAAAAHNvdW4AAAAAAAAAAAAAAABTb3VuZEhhbmRsZXIAAAABYG1pbmYAAAAQc21oZAAAAAAAAAAAAAAAJGRpbmYA",
    "AAAcZHJlZgAAAAAAAAABAAAADHVybCAAAAABAAABJHN0YmwAAABqc3RzZAAAAAAAAAABAAAAWm1wNGEAAAAAAAAAAQAAAAAA",
    "AAAAAAEAEAAAAAC7gAAAAAAANmVzZHMAAAAAA4CAgCUAAQAEgICAF0AVAAAAAAEXtAABF7QFgICABRGIVuUABoCAgAECAAAA",
    "IHN0dHMAAAAAAAAAAgAAAAUAAAQAAAAAAQAAAsAAAAAcc3RzYwAAAAAAAAABAAAAAQAAAAYAAAABAAAALHN0c3oAAAAAAAAA",
    "AAAAAAYAAAD5AAAA5AAAAIIAAACtAAAAlQAAAJ0AAAAUc3RjbwAAAAAAAAABAAAALAAAABpzZ3BkAQAAAHJvbGwAAAACAAAA",
    "Af//AAAAHHNiZ3AAAAAAcm9sbAAAAAEAAAAGAAAAAQAAAGJ1ZHRhAAAAWm1ldGEAAAAAAAAAIWhkbHIAAAAAAAAAAG1kaXJh",
    "cHBsAAAAAAAAAAAAAAAALWlsc3QAAAAlqXRvbwAAAB1kYXRhAAAAAQAAAABMYXZmNjAuMTYuMTAw",
);

const LEADING_EMPTY_NO_EDIT: &str = concat!(
    "AAAAHGZ0eXBNNEEgAAACAE00QSBpc29taXNvMgAAAAhmcmVlAAAERm1kYXTeAgBMYXZjNjAuMzEuMTAyAAJgqlIstKYqL0JB",
    "0KuPn2VU57+vMsvEk3JJNxCB2SBs1bNcqzk0ZxVhxWNs2Ns2U5i3DZXTOciAB/oYz3Da341HdQ6+1ri3VeG4HXuO8NtO3Y3F",
    "Y3FXGerMdicdYa1O1pq2UtlKpSqUqlKpSqUqlKpSqUqlJpSaUqlKoyqMmpJpGkZ2dpJnmdnZ2kaRnZ2dnaRpGdnZ2dnmdnZ2",
    "dnZ20sDAwMDIpZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZYoo",
    "ooooouABGpja2j3WYiYTZiJhN1MJf9P864jy1etR//F/3/Wya1rP4//i/7/rZNa1n2//i/5+9l641yIy7Rkc6tChISEtPnru",
    "jZMRn66SknsnUzcT6apwe9fs5G2qbz+GCqZ/yy+Ne/5ITLLevBITKqxwkJlle87hISEqld3cqEhMsqr2bhISEwEAD567o2TE",
    "Z+ukpJ7J1M3E+mqcHvX7ORtqm8/hgqmf8svjXv+SEyy3rwSEyqscJCZZXvO4SEhKpXd3KhITLKq9m4SEhMBABeNjlcK98L78",
    "8vC//LLLLLLLLKDSyopZYDgA9vUtlFYZD2AhoIhf2+3nit7yf2/63rSpbdyDSWkkkkXVf/f/6DVuTRMLSKKlSrIjFGUZYlT2",
    "JW1mLsroN3y0Usb6l693Vxd+261lUcuFkZ3acpsLUkZk2bNjRSFXhhhhIJ0444zMhe/voHx8Zh7++iABEy1atWqklAZk2bNj",
    "QQLnAPQ1GJaqMg9DQdHYdEIQC4QE4QDIgC4Tjx96xv/0/1///e1pWl/aR5Xcl3X8fFc8A7O2GLs7YNBYyjGnsR3FiOkjdE5Z",
    "wASQqpktlhd3GuxtI5i412NmHEtp2rOcVnOKxNmnZ6RbYmzWWenZ6dnpGSatp2estHrdiBgYGBgYGBgYGBgYEIEIEIE7tvJm",
    "gs9WLjDTDQAACIAAR3vOAZ98AAAA1/wtgAAAI/E4QcABEjUttGYqC1xB0Lmp35TPf+v/zkzWRktNx+JJIkQ+g+Io3SchyJij",
    "hz+63r1i8OZ/ydtZMF4J6nmPnrcmChezCEVn2kmQ1Ti1Z+Qq3cfudF6poFHla6ybZA7XTvE4TJx8ycWM48poodty7M3ynPOq",
    "7Js5bcqLM3ynPMbp2slLJLJXynMwHwUT4BHSMB8GGMfhJXd2cAEkNS30lg6Fg68TFNvX4cXmdKripj7u4kkREBhvdL84y5uk",
    "v7jsnDu6o37d5trz7lsrPP1GqMnJycIkMX76NsFDdobI8629p+xY1393lD9w9U192tSWR4yozqOmzq3JcaV5zjYLaoau2ExZ",
    "n5LOrWUcG1UjNnMjYTE8/JZ2UybBtVIzZyxagNlwrVwyVRhJTilch41yHjNXgNMxYcAAAALvbW9vdgAAAGxtdmhkAAAAAAAA",
    "AAAAAAAAAAAD6AAAAHoAAQAAAQAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAA",
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAgAAAhl0cmFrAAAAXHRraGQAAAADAAAAAAAAAAAAAAABAAAAAAAAAHoAAAAAAAAAAAAA",
    "AAEBAAAAAAEAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAG1bWRpYQAAACBtZGhkAAAAAAAA",
    "AAAAAAAAAAC7gAAAFsBVxAAAAAAALWhkbHIAAAAAAAAAAHNvdW4AAAAAAAAAAAAAAABTb3VuZEhhbmRsZXIAAAABYG1pbmYA",
    "AAAQc21oZAAAAAAAAAAAAAAAJGRpbmYAAAAcZHJlZgAAAAAAAAABAAAADHVybCAAAAABAAABJHN0YmwAAABqc3RzZAAAAAAA",
    "AAABAAAAWm1wNGEAAAAAAAAAAQAAAAAAAAAAAAEAEAAAAAC7gAAAAAAANmVzZHMAAAAAA4CAgCUAAQAEgICAF0AVAAAAAAEX",
    "tAABF7QFgICABRGIVuUABoCAgAECAAAAIHN0dHMAAAAAAAAAAgAAAAUAAAQAAAAAAQAAAsAAAAAcc3RzYwAAAAAAAAABAAAA",
    "AQAAAAYAAAABAAAALHN0c3oAAAAAAAAAAAAAAAYAAAD5AAAA5AAAAIIAAACtAAAAlQAAAJ0AAAAUc3RjbwAAAAAAAAABAAAA",
    "LAAAABpzZ3BkAQAAAHJvbGwAAAACAAAAAf//AAAAHHNiZ3AAAAAAcm9sbAAAAAEAAAAGAAAAAQAAAGJ1ZHRhAAAAWm1ldGEA",
    "AAAAAAAAIWhkbHIAAAAAAAAAAG1kaXJhcHBsAAAAAAAAAAAAAAAALWlsc3QAAAAlqXRvbwAAAB1kYXRhAAAAAQAAAABMYXZm",
    "NjAuMTYuMTAw",
);

const OVERSIZED_V1_EDIT_LIST: &str = concat!(
    "AAAAHGZ0eXBNNEEgAAACAE00QSBpc29taXNvMgAAAAhmcmVlAAAERm1kYXTeAgBMYXZjNjAuMzEuMTAyAAJgqlIstKYqL0JB",
    "0KuPn2VU57+vMsvEk3JJNxCB2SBs1bNcqzk0ZxVhxWNs2Ns2U5i3DZXTOciAB/oYz3Da341HdQ6+1ri3VeG4HXuO8NtO3Y3F",
    "Y3FXGerMdicdYa1O1pq2UtlKpSqUqlKpSqUqlKpSqUqlJpSaUqlKoyqMmpJpGkZ2dpJnmdnZ2kaRnZ2dnaRpGdnZ2dnmdnZ2",
    "dnZ20sDAwMDIpZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZYoo",
    "ooooouABGpja2j3WYiYTZiJhN1MJf9P864jy1etR//F/3/Wya1rP4//i/7/rZNa1n2//i/5+9l641yIy7Rkc6tChISEtPnru",
    "jZMRn66SknsnUzcT6apwe9fs5G2qbz+GCqZ/yy+Ne/5ITLLevBITKqxwkJlle87hISEqld3cqEhMsqr2bhISEwEAD567o2TE",
    "Z+ukpJ7J1M3E+mqcHvX7ORtqm8/hgqmf8svjXv+SEyy3rwSEyqscJCZZXvO4SEhKpXd3KhITLKq9m4SEhMBABeNjlcK98L78",
    "8vC//LLLLLLLLKDSyopZYDgA9vUtlFYZD2AhoIhf2+3nit7yf2/63rSpbdyDSWkkkkXVf/f/6DVuTRMLSKKlSrIjFGUZYlT2",
    "JW1mLsroN3y0Usb6l693Vxd+261lUcuFkZ3acpsLUkZk2bNjRSFXhhhhIJ0444zMhe/voHx8Zh7++iABEy1atWqklAZk2bNj",
    "QQLnAPQ1GJaqMg9DQdHYdEIQC4QE4QDIgC4Tjx96xv/0/1///e1pWl/aR5Xcl3X8fFc8A7O2GLs7YNBYyjGnsR3FiOkjdE5Z",
    "wASQqpktlhd3GuxtI5i412NmHEtp2rOcVnOKxNmnZ6RbYmzWWenZ6dnpGSatp2estHrdiBgYGBgYGBgYGBgYEIEIEIE7tvJm",
    "gs9WLjDTDQAACIAAR3vOAZ98AAAA1/wtgAAAI/E4QcABEjUttGYqC1xB0Lmp35TPf+v/zkzWRktNx+JJIkQ+g+Io3SchyJij",
    "hz+63r1i8OZ/ydtZMF4J6nmPnrcmChezCEVn2kmQ1Ti1Z+Qq3cfudF6poFHla6ybZA7XTvE4TJx8ycWM48poodty7M3ynPOq",
    "7Js5bcqLM3ynPMbp2slLJLJXynMwHwUT4BHSMB8GGMfhJXd2cAEkNS30lg6Fg68TFNvX4cXmdKripj7u4kkREBhvdL84y5uk",
    "v7jsnDu6o37d5trz7lsrPP1GqMnJycIkMX76NsFDdobI8629p+xY1393lD9w9U192tSWR4yozqOmzq3JcaV5zjYLaoau2ExZ",
    "n5LOrWUcG1UjNnMjYTE8/JZ2UybBtVIzZyxagNlwrVwyVRhJTilch41yHjNXgNMxYcAAAANHbW9vdgAAAHhtdmhkAQAAAAAA",
    "AAAAAAAAAAAAAAAAAAAAAAPoAAAAASoF8mQAAQAAAQAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAA",
    "AABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgAAAmV0cmFrAAAAaHRraGQBAAADAAAAAAAAAAAAAAAAAAAAAAAA",
    "AAEAAAAAAAAAASoF8mQAAAAAAAAAAAAAAAEBAAAAAAEAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAA",
    "AAAAAABAZWR0cwAAADhlbHN0AQAAAAAAAAIAAAABKgXx6v//////////AAEAAAAAAAAAAAB6AAAAAAAAAAAAAQAAAAABtW1k",
    "aWEAAAAgbWRoZAAAAAAAAAAAAAAAAAAAu4AAABbAVcQAAAAAAC1oZGxyAAAAAAAAAABzb3VuAAAAAAAAAAAAAAAAU291bmRI",
    "YW5kbGVyAAAAAWBtaW5mAAAAEHNtaGQAAAAAAAAAAAAAACRkaW5mAAAAHGRyZWYAAAAAAAAAAQAAAAx1cmwgAAAAAQAAASRz",
    "dGJsAAAAanN0c2QAAAAAAAAAAQAAAFptcDRhAAAAAAAAAAEAAAAAAAAAAAABABAAAAAAu4AAAAAAADZlc2RzAAAAAAOAgIAl",
    "AAEABICAgBdAFQAAAAABF7QAARe0BYCAgAURiFblAAaAgIABAgAAACBzdHRzAAAAAAAAAAIAAAAFAAAEAAAAAAEAAALAAAAA",
    "HHN0c2MAAAAAAAAAAQAAAAEAAAAGAAAAAQAAACxzdHN6AAAAAAAAAAAAAAAGAAAA+QAAAOQAAACCAAAArQAAAJUAAACdAAAA",
    "FHN0Y28AAAAAAAAAAQAAACwAAAAac2dwZAEAAAByb2xsAAAAAgAAAAH//wAAABxzYmdwAAAAAHJvbGwAAAABAAAABgAAAAEA",
    "AABidWR0YQAAAFptZXRhAAAAAAAAACFoZGxyAAAAAAAAAABtZGlyYXBwbAAAAAAAAAAAAAAAAC1pbHN0AAAAJal0b28AAAAd",
    "ZGF0YQAAAAEAAAAATGF2ZjYwLjE2LjEwMA==",
);

struct EmbeddedFixture {
    name: &'static str,
    encoded: &'static str,
    byte_len: usize,
    sha256: &'static str,
}

const MONO_TIMED_FIXTURE: EmbeddedFixture = EmbeddedFixture {
    name: "mono-edit-list.m4a",
    encoded: MONO_EDIT_LIST,
    byte_len: 2_013,
    sha256: "6ce53d7289bd411d52175946604a4bf4a2f23aec703f3e3db249b269f3f91fd3",
};

const MONO_CONTROL_FIXTURE: EmbeddedFixture = EmbeddedFixture {
    name: "mono-no-edit.m4a",
    encoded: MONO_NO_EDIT,
    byte_len: 1_977,
    sha256: "84f09bfd4e685778b02f482f0a5c72145010cc3d0d0874de39068fa3c1d09e12",
};

const STEREO_TIMED_FIXTURE: EmbeddedFixture = EmbeddedFixture {
    name: "stereo-edit-list.m4a",
    encoded: STEREO_EDIT_LIST,
    byte_len: 1_273,
    sha256: "b69253693ac5770b1c7752451b10eb3cce542900169f49911bc6a97ff84be453",
};

const STEREO_CONTROL_FIXTURE: EmbeddedFixture = EmbeddedFixture {
    name: "stereo-no-edit.m4a",
    encoded: STEREO_NO_EDIT,
    byte_len: 1_237,
    sha256: "19a20cce9fa70da426fd0f3b25d0e78a1b3047eadbcad5effc6db43dbc22c8fd",
};

const LEADING_EMPTY_FIXTURE: EmbeddedFixture = EmbeddedFixture {
    name: "leading-empty-250ms.m4a",
    encoded: LEADING_EMPTY_EDIT_LIST,
    byte_len: 1_929,
    sha256: "94c3f9ec0a15f6f45d5b37c7fb3f838b9f4db54d2534df6485621e7db2f3fd8a",
};

const LEADING_EMPTY_CONTROL_FIXTURE: EmbeddedFixture = EmbeddedFixture {
    name: "leading-empty-no-edit.m4a",
    encoded: LEADING_EMPTY_NO_EDIT,
    byte_len: 1_881,
    sha256: "d9e483fd8a5e5acc3be5b45a92b484543b550dca154d359e33ec0824e7a5de57",
};

const OVERSIZED_V1_FIXTURE: EmbeddedFixture = EmbeddedFixture {
    name: "oversized-v1-edit-list.m4a",
    encoded: OVERSIZED_V1_EDIT_LIST,
    byte_len: 1_969,
    sha256: "fb715deb3f56ce4814a54892b0e410b3bf1c2881c88c04ba28ca1f9ca2bc283f",
};

fn fixture_bytes(fixture: &EmbeddedFixture) -> Vec<u8> {
    let bytes = STANDARD
        .decode(fixture.encoded)
        .unwrap_or_else(|error| panic!("decode embedded {}: {error}", fixture.name));
    assert_eq!(
        bytes.len(),
        fixture.byte_len,
        "{} byte length",
        fixture.name
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(&bytes)),
        fixture.sha256,
        "{} SHA-256",
        fixture.name
    );
    bytes
}

fn write_fixture(directory: &tempfile::TempDir, fixture: &EmbeddedFixture) -> std::path::PathBuf {
    let path = directory.path().join(fixture.name);
    std::fs::write(&path, fixture_bytes(fixture))
        .unwrap_or_else(|error| panic!("write embedded {}: {error}", fixture.name));
    path
}

fn unique_box_type_offset(bytes: &[u8], name: &[u8; 4]) -> usize {
    let offsets = bytes
        .windows(4)
        .enumerate()
        .filter_map(|(offset, candidate)| (candidate == name).then_some(offset))
        .collect::<Vec<_>>();
    assert_eq!(offsets.len(), 1, "fixture must contain one {name:?} box");
    offsets[0]
}

fn small_v1_edit_fixture_bytes() -> Vec<u8> {
    let mut bytes = fixture_bytes(&OVERSIZED_V1_FIXTURE);
    let mvhd = unique_box_type_offset(&bytes, b"mvhd");
    let tkhd = unique_box_type_offset(&bytes, b"tkhd");
    let elst = unique_box_type_offset(&bytes, b"elst");

    assert_eq!(&bytes[mvhd + 4..mvhd + 8], &[1, 0, 0, 0]);
    let movie_duration = mvhd + 28;
    assert_eq!(
        &bytes[movie_duration..movie_duration + 8],
        &5_000_000_100u64.to_be_bytes()
    );
    assert_eq!(&bytes[tkhd + 4..tkhd + 8], &[1, 0, 0, 3]);
    let track_duration = tkhd + 32;
    assert_eq!(
        &bytes[track_duration..track_duration + 8],
        &5_000_000_100u64.to_be_bytes()
    );
    assert_eq!(&bytes[elst + 4..elst + 8], &[1, 0, 0, 0]);
    assert_eq!(&bytes[elst + 8..elst + 12], &2u32.to_be_bytes());
    let first_duration = elst + 12;
    assert_eq!(
        &bytes[first_duration..first_duration + 8],
        &4_999_999_978u64.to_be_bytes()
    );

    // Keep all three full boxes at version 1 while making the movie and track
    // duration metadata consistent with the patched 228 + 122 time-unit edits.
    bytes[movie_duration..movie_duration + 8].copy_from_slice(&350u64.to_be_bytes());
    bytes[track_duration..track_duration + 8].copy_from_slice(&350u64.to_be_bytes());
    bytes[first_duration..first_duration + 8].copy_from_slice(&228u64.to_be_bytes());
    assert_eq!(
        format!("{:x}", Sha256::digest(&bytes)),
        "98d8b4b94bd417ab94160ddb0843f1e53972880c632326af208f3d308a36bca4",
        "patched small version-1 fixture SHA-256"
    );
    bytes
}

fn write_fixture_bytes(
    directory: &tempfile::TempDir,
    name: &str,
    bytes: &[u8],
) -> std::path::PathBuf {
    let path = directory.path().join(name);
    std::fs::write(&path, bytes).unwrap_or_else(|error| panic!("write {name}: {error}"));
    path
}

fn assert_aac_audio_probe(path: &std::path::Path) {
    let probe = probe_file(path).expect("probe timed M4A fixture");
    assert_eq!(probe.format, AudioFormat::M4a);
    assert_eq!(probe.codec, AudioCodec::Aac);
    assert_eq!(probe.audio_tracks, 1);
    assert!(!probe.has_non_audio_tracks);
}

fn rms(samples: &[f64]) -> f64 {
    (samples.iter().map(|sample| sample * sample).sum::<f64>() / samples.len() as f64).sqrt()
}

fn assert_head_and_tail_energy(decoded: &DecodedPcm) {
    const WINDOW: usize = 256;
    for channel in &decoded.channels {
        assert!(rms(&channel[..WINDOW]) > 0.01, "silent decoded head");
        assert!(
            rms(&channel[channel.len() - WINDOW..]) > 0.01,
            "silent decoded tail"
        );
    }
}

fn collect_bounded_stream(path: &std::path::Path, block_frames: usize) -> Vec<Vec<f64>> {
    let session = AudioInputSession::open(path).expect("open M4A stream session");
    let reader = AudioStreamReader::from_session(session, DecodeLimits::default())
        .expect("open bounded M4A reader");
    collect_bounded_reader(reader, block_frames)
}

fn collect_bounded_reader(mut reader: AudioStreamReader, block_frames: usize) -> Vec<Vec<f64>> {
    assert_eq!(reader.info().format, AudioFormat::M4a);
    assert_eq!(reader.info().codec, AudioCodec::Aac);
    let mut output = vec![Vec::new(); reader.info().channels()];
    while let Some(block) = reader
        .next_block(block_frames)
        .expect("decode bounded M4A block")
    {
        assert!(block[0].len() <= block_frames);
        for (destination, source) in output.iter_mut().zip(block) {
            destination.extend(source);
        }
    }
    output
}

#[test]
fn mono_edit_list_selects_the_exact_aac_presentation_span() {
    let directory = tempfile::tempdir().expect("create mono M4A fixture directory");
    let timed_path = write_fixture(&directory, &MONO_TIMED_FIXTURE);
    let control_path = write_fixture(&directory, &MONO_CONTROL_FIXTURE);

    assert_aac_audio_probe(&timed_path);
    let timed = decode_file(&timed_path).expect("decode mono edit-list fixture");
    let control = decode_file(&control_path).expect("decode mono no-edit control");

    assert_eq!(timed.sample_rate, 44_100);
    assert_eq!(control.sample_rate, 44_100);
    assert_eq!(timed.channel_layout(), ChannelLayout::Mono);
    assert_eq!(control.channel_layout(), ChannelLayout::Mono);
    assert_eq!(control.frames(), 6_144);
    assert_eq!(timed.frames(), 4_410);
    assert_eq!(
        timed.channels[0].as_slice(),
        &control.channels[0][1_024..5_434]
    );
    assert_head_and_tail_energy(&timed);
}

#[test]
fn stereo_edit_list_scales_movie_time_and_selects_the_exact_pcm_span() {
    let directory = tempfile::tempdir().expect("create stereo M4A fixture directory");
    let timed_path = write_fixture(&directory, &STEREO_TIMED_FIXTURE);
    let control_path = write_fixture(&directory, &STEREO_CONTROL_FIXTURE);

    assert_aac_audio_probe(&timed_path);
    let timed = decode_file(&timed_path).expect("decode stereo edit-list fixture");
    let control = decode_file(&control_path).expect("decode stereo no-edit control");

    assert_eq!(timed.sample_rate, 48_000);
    assert_eq!(control.sample_rate, 48_000);
    assert_eq!(timed.channel_layout(), ChannelLayout::Stereo);
    assert_eq!(control.channel_layout(), ChannelLayout::Stereo);
    assert_eq!(control.frames(), 2_048);
    assert_eq!(timed.frames(), 1_007);
    for (timed_channel, control_channel) in timed.channels.iter().zip(&control.channels) {
        assert_eq!(timed_channel.as_slice(), &control_channel[1_024..2_031]);
    }
    assert_head_and_tail_energy(&timed);
}

#[test]
fn bounded_aac_stream_matches_whole_decode_for_edit_and_no_edit_timelines() {
    let directory = tempfile::tempdir().expect("create bounded M4A fixture directory");
    for (timed, control, block_frames) in [
        (&MONO_TIMED_FIXTURE, &MONO_CONTROL_FIXTURE, 257),
        (&STEREO_TIMED_FIXTURE, &STEREO_CONTROL_FIXTURE, 113),
        (&LEADING_EMPTY_FIXTURE, &LEADING_EMPTY_CONTROL_FIXTURE, 509),
    ] {
        for fixture in [timed, control] {
            let path = write_fixture(&directory, fixture);
            let whole = decode_file(&path).expect("decode whole M4A fixture");
            let streamed = collect_bounded_stream(&path, block_frames);
            assert_eq!(streamed, whole.channels, "{}", fixture.name);
        }
    }
}

#[test]
fn bounded_aac_stream_decoder_allowance_has_an_exact_limit_boundary() {
    let directory = tempfile::tempdir().expect("create M4A budget fixture directory");
    let path = write_fixture(&directory, &STEREO_TIMED_FIXTURE);
    let mut session = AudioInputSession::open(&path).expect("open M4A budget session");
    let info = inspect_audio_stream_session(&mut session, DecodeLimits::default())
        .expect("inspect uncapped M4A stream");
    let exact =
        DecodeLimits::default().with_max_working_set_bytes(Some(info.decoder_additional_bytes));
    let mut exact_session = AudioInputSession::open(&path).expect("open exact M4A budget session");
    inspect_audio_stream_session(&mut exact_session, exact)
        .expect("accept exact M4A decoder allowance");

    let mut short_session = AudioInputSession::open(&path).expect("open short M4A budget session");
    let error = inspect_audio_stream_session(
        &mut short_session,
        DecodeLimits::default().with_max_working_set_bytes(Some(
            info.decoder_additional_bytes
                .checked_sub(1)
                .expect("M4A allowance is nonzero"),
        )),
    )
    .expect_err("reject one byte below the M4A decoder allowance");
    assert!(error.contains("M4A/AAC stream decoder"), "{error}");
    assert!(error.contains("working-set limit"), "{error}");
}

#[test]
fn bounded_aac_stream_keeps_using_the_opened_inode_after_path_replacement() {
    let directory = tempfile::tempdir().expect("create M4A inode fixture directory");
    let path = write_fixture(&directory, &MONO_TIMED_FIXTURE);
    let moved = directory.path().join("opened-original.m4a");
    let session = AudioInputSession::open(&path).expect("open original M4A session");
    std::fs::rename(&path, &moved).expect("move opened M4A pathname");
    std::fs::write(&path, fixture_bytes(&STEREO_CONTROL_FIXTURE))
        .expect("write replacement M4A pathname");

    let reader = AudioStreamReader::from_session(session, DecodeLimits::default())
        .expect("open held M4A inode as a bounded stream");
    let streamed = collect_bounded_reader(reader, 211);
    let original = decode_file(&moved).expect("decode moved original M4A");
    assert_eq!(streamed, original.channels);
}

#[test]
fn leading_empty_edit_inserts_the_exact_silence_before_media() {
    const SILENT_FRAMES: usize = 10_944;
    const TIMED_MEDIA_FRAMES: usize = 5_824;
    const MOVIE_ROUNDING_FRAMES: usize = 32;
    const PRESENTATION_MEDIA_FRAMES: usize = TIMED_MEDIA_FRAMES + MOVIE_ROUNDING_FRAMES;

    let directory = tempfile::tempdir().expect("create leading-empty M4A fixture directory");
    let timed_path = write_fixture(&directory, &LEADING_EMPTY_FIXTURE);
    let control_path = write_fixture(&directory, &LEADING_EMPTY_CONTROL_FIXTURE);

    assert_aac_audio_probe(&timed_path);
    let decoded = decode_file(&timed_path).expect("decode leading-empty edit-list fixture");
    let control = decode_file(&control_path).expect("decode leading-empty no-edit control");

    assert_eq!(decoded.sample_rate, 48_000);
    assert_eq!(control.sample_rate, 48_000);
    assert_eq!(decoded.channel_layout(), ChannelLayout::Mono);
    assert_eq!(control.channel_layout(), ChannelLayout::Mono);
    assert_eq!(control.frames(), 6_144);
    assert_eq!(decoded.frames(), 16_800);
    assert!(decoded.channels[0][..SILENT_FRAMES]
        .iter()
        .all(|sample| *sample == 0.0));
    assert_eq!(decoded.n_channels(), control.n_channels());
    for (decoded_channel, control_channel) in decoded.channels.iter().zip(&control.channels) {
        assert_eq!(
            &decoded_channel[SILENT_FRAMES..SILENT_FRAMES + TIMED_MEDIA_FRAMES],
            &control_channel[..TIMED_MEDIA_FRAMES]
        );
        assert!(decoded_channel
            [SILENT_FRAMES + TIMED_MEDIA_FRAMES..SILENT_FRAMES + PRESENTATION_MEDIA_FRAMES]
            .iter()
            .all(|sample| *sample == 0.0));
    }
    // FFmpeg rounds the remaining 5,824 media frames up to a 122-ms movie
    // timescale entry. Preserve that declared presentation length, but fill
    // the 32-frame quantization remainder with silence instead of exposing
    // untimed AAC decoder padding.
    // The AAC packets retain their 1,024-frame encoder-delay prefix because
    // this edit begins at media time zero. Check energy after that prefix so
    // the assertion distinguishes the intentional empty edit from codec delay.
    assert!(rms(&decoded.channels[0][SILENT_FRAMES + 1_024..SILENT_FRAMES + 1_280]) > 0.01);
    assert!(rms(&decoded.channels[0][decoded.frames() - 256..]) > 0.01);
}

#[test]
fn oversized_v1_edit_list_is_rejected_before_output_allocation() {
    let directory = tempfile::tempdir().expect("create oversized v1 M4A fixture directory");
    let path = write_fixture(&directory, &OVERSIZED_V1_FIXTURE);

    for attempt in 1..=2 {
        let result = std::panic::catch_unwind(|| decode_file(&path));
        let error = result
            .unwrap_or_else(|_| panic!("oversized v1 edit-list attempt {attempt} panicked"))
            .expect_err("oversized v1 edit list unexpectedly decoded");
        assert!(
            error.contains("working set requires 1920000087552 bytes"),
            "unexpected oversized edit error on attempt {attempt}: {error}"
        );
        assert!(
            error.contains("limit is 536870912 bytes"),
            "missing 512 MiB limit on attempt {attempt}: {error}"
        );
    }
}

#[test]
fn patched_small_v1_edit_list_runs_the_full_public_decode_path() {
    let directory = tempfile::tempdir().expect("create small v1 M4A fixture directory");
    let v1_path = write_fixture_bytes(
        &directory,
        "small-v1-edit-list.m4a",
        &small_v1_edit_fixture_bytes(),
    );
    let v0_path = write_fixture(&directory, &LEADING_EMPTY_FIXTURE);

    assert_aac_audio_probe(&v1_path);
    let v1 = decode_file(&v1_path).expect("decode patched small version-1 edit list");
    let v0 = decode_file(&v0_path).expect("decode equivalent version-0 edit list");

    assert_eq!(v1.sample_rate, 48_000);
    assert_eq!(v1.channel_layout(), ChannelLayout::Mono);
    assert_eq!(v1.frames(), 16_800);
    assert_eq!(v1.channels, v0.channels);
}
