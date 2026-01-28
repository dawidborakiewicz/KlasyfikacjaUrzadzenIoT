import pandas as pd

def add_ip_proto_flags(df: pd.DataFrame, col: str = "ip.proto") -> pd.DataFrame:
    # zamiana na liczby; puste -> NaN
    proto = pd.to_numeric(df.get(col), errors="coerce")

    df["is_icmp"] = (proto == 1).astype("int8")
    df["is_igmp"] = (proto == 2).astype("int8")
    df["is_tcp"]  = (proto == 6).astype("int8")
    df["is_udp"]  = (proto == 17).astype("int8")

    df.drop(columns="ip.proto", inplace=True)

    return df

def add_frame_protocol_flags(
    df: pd.DataFrame,
    col: str = "frame.protocols",
) -> pd.DataFrame:
    """
    frame.protocols zwykle wygląda jak 'eth:ethertype:ip:tcp:tls' itp.
    Robimy flagi: is_ssdp, is_tls, is_dns, is_icmp.
    """
    if col not in df.columns:
        for name in ["is_ssdp", "is_tls", "is_dns", "is_icmp"]:
            df[name] = 0
        return df

    protos = (
        df[col]
        .astype("string")
        .str.lower()
        .fillna("")
        .str.split(":")
    )

    # protos jest serią list; sprawdzamy membership
    df["is_ssdp"] = protos.apply(lambda xs: int("ssdp" in xs))
    df["is_tls"]  = protos.apply(lambda xs: int("tls" in xs or "ssl" in xs))
    df["is_dns"]  = protos.apply(lambda xs: int("dns" in xs))
    df["is_icmp"] = protos.apply(lambda xs: int("icmp" in xs))

    # małe typy
    for c in ["is_ssdp", "is_tls", "is_dns", "is_icmp"]:
        df[c] = df[c].astype("int8")

    df.drop(columns="frame.protocols", inplace=True)

    return df

def add_tcp_flags(
    df: pd.DataFrame,
    col: str = "tcp.flags",
    prefix: str = "tcp",
) -> pd.DataFrame:
    """
    tcp.flags zwykle jest stringiem '0x0012' itp.
    Rozbijamy na 6 flag: fin, syn, rst, psh, ack, urg.
    Bity TCP:
      FIN=0x01 SYN=0x02 RST=0x04 PSH=0x08 ACK=0x10 URG=0x20
    """
    out_cols = [f"{prefix}_{x}" for x in ["fin", "syn", "rst", "psh", "ack", "urg"]]
    for c in out_cols:
        if c not in df.columns:
            df[c] = 0

    if col not in df.columns:
        return df

    s = df[col].astype("string").str.strip().str.lower()

    # parse hex ("0x0012") lub decimal ("18") jeśli kiedyś się trafi
    def parse_flags(x: str):
        if x is None:
            return pd.NA
        x = str(x).strip().lower()
        if not x or x == "<na>":
            return pd.NA
        try:
            if x.startswith("0x"):
                return int(x, 16)
            return int(x)
        except ValueError:
            return pd.NA

    flags = s.map(parse_flags)

    # bitmask
    df[f"{prefix}_fin"] = ((flags.fillna(0).astype("int64") & 0x01) != 0).astype("int8")
    df[f"{prefix}_syn"] = ((flags.fillna(0).astype("int64") & 0x02) != 0).astype("int8")
    df[f"{prefix}_rst"] = ((flags.fillna(0).astype("int64") & 0x04) != 0).astype("int8")
    df[f"{prefix}_psh"] = ((flags.fillna(0).astype("int64") & 0x08) != 0).astype("int8")
    df[f"{prefix}_ack"] = ((flags.fillna(0).astype("int64") & 0x10) != 0).astype("int8")
    df[f"{prefix}_urg"] = ((flags.fillna(0).astype("int64") & 0x20) != 0).astype("int8")

    df.drop(columns="tcp.flags", inplace=True)
    return df


def one_hot(df: pd.DataFrame):
    df = add_ip_proto_flags(df)
    df = add_tcp_flags(df)
    df = add_frame_protocol_flags(df)
    return df