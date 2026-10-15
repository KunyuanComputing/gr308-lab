set(GRSERVE_SANITIZERS "" CACHE STRING "Comma-separated sanitizer list: address,undefined or thread")
option(GRSERVE_SANITIZE_THIRD_PARTY
  "Instrument pinned third-party dependencies with enabled sanitizers" ON)

function(_grserve_add_sanitizer_options)
  add_compile_options(${_grserve_common_sanitize_flags})
  add_link_options(${_grserve_common_sanitize_flags})
endfunction()

function(grserve_enable_project_sanitizers)
  if(NOT GRSERVE_SANITIZE_THIRD_PARTY AND NOT GRSERVE_SANITIZERS STREQUAL "")
    _grserve_add_sanitizer_options()
  endif()
endfunction()

if(GRSERVE_SANITIZERS STREQUAL "")
  # Sanitizers disabled.
elseif(NOT GRSERVE_SANITIZERS STREQUAL "address,undefined" AND
       NOT GRSERVE_SANITIZERS STREQUAL "thread")
  message(FATAL_ERROR
    "Unsupported GRSERVE_SANITIZERS='${GRSERVE_SANITIZERS}'; "
    "expected empty, address,undefined, or thread")
else()
  string(REPLACE "," ";" _grserve_sanitizers "${GRSERVE_SANITIZERS}")
  set(_grserve_sanitize_flags "")
  foreach(_sanitizer IN LISTS _grserve_sanitizers)
    if(_sanitizer STREQUAL "address" OR _sanitizer STREQUAL "undefined" OR _sanitizer STREQUAL "thread")
      list(APPEND _grserve_sanitize_flags "${_sanitizer}")
    endif()
  endforeach()
  list(JOIN _grserve_sanitize_flags "," _grserve_sanitize_value)
  set(_grserve_common_sanitize_flags
    -fno-omit-frame-pointer
    "-fsanitize=${_grserve_sanitize_value}")
  if("undefined" IN_LIST _grserve_sanitizers)
    list(APPEND _grserve_common_sanitize_flags -fno-sanitize-recover=undefined)
  endif()
  if(GRSERVE_SANITIZE_THIRD_PARTY)
    # Called before Dependencies.cmake so pinned dependencies and project targets
    # use one compatible sanitizer runtime. Set the option OFF to instrument only
    # the GRServe targets declared after grserve_enable_project_sanitizers().
    _grserve_add_sanitizer_options()
  endif()
endif()
